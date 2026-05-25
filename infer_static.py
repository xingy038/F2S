# infer_human_batch.py
import os
import cv2
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from model.human import Human
from vis.utils import save_normal_map
from utils.dinov3_helper import get_model_config, load_model_checkpoint
import random


# -------- Preprocessing & Restoration --------
def preprocess_smart(img_path: Path, target: int | None, size_divisor: int = 16):
    assert size_divisor > 0, "size_divisor must be positive."

    img = np.asarray(Image.open(img_path).convert("RGB"))
    h0, w0 = img.shape[:2]

    if target is not None:
        s = min(target / h0, target / w0)
        new_h, new_w = int(round(h0 * s)), int(round(w0 * s))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        img_res = cv2.resize(img, (new_w, new_h), interpolation=interp)

        pad_left  = (target - new_w) // 2
        pad_right = target - new_w - pad_left
        pad_top   = (target - new_h) // 2
        pad_bottom= target - new_h - pad_top
        canvas = target
    else:
        L0 = max(h0, w0)
        L = int(round(L0 / size_divisor) * size_divisor)
        L = max(L, size_divisor)

        s = L / float(L0)
        new_h, new_w = int(round(h0 * s)), int(round(w0 * s))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        img_res = cv2.resize(img, (new_w, new_h), interpolation=interp)

        pad_left = (L - new_w) // 2
        pad_right = L - new_w - pad_left
        pad_top = (L - new_h) // 2
        pad_bottom = L - new_h - pad_top
        canvas = L

    img_pad = cv2.copyMakeBorder(
        img_res, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE
    )

    img_f32 = img_pad.astype(np.float32) / 255.0  # 0-1
    img_t = torch.from_numpy(img_f32).permute(2, 0, 1).contiguous()

    meta = {
        "pad": (pad_left, pad_top),        # (x, y)
        "orig_size": (h0, w0),
        "resized_size": (new_h, new_w),
        "canvas": canvas
    }
    return img_t, meta


def unpad_and_restore(x: torch.Tensor, meta: dict, mode="bilinear"):
    """Remove padding and resize back to the original image size. x: [B,C,S,S] or [C,S,S]."""
    batched = (x.dim() == 4)
    if not batched:
        x = x.unsqueeze(0)
    pad_x, pad_y = meta["pad"]
    S = x.shape[-1]
    h_sc = S - 2 * pad_y
    w_sc = S - 2 * pad_x
    x = x[..., pad_y: pad_y + h_sc, pad_x: pad_x + w_sc]
    h0, w0 = meta["orig_size"]
    x = F.interpolate(x, size=(h0, w0), mode=mode, align_corners=True)
    return x if batched else x[0]


# -------- Saving utilities --------
def save_depth_masked(pred_depth: torch.Tensor, valid_mask: torch.Tensor, output_path: str):
    d = pred_depth
    if d.dim() == 3:
        d = d.squeeze(0)
    m = valid_mask
    if m.dim() == 3:
        m = m.squeeze(0)
    m = m.bool()

    depth_np = d.detach().cpu().numpy().astype(np.float32)
    mask_np  = m.detach().cpu().numpy()

    valid_vals = depth_np[mask_np]
    if valid_vals.size == 0:
        print(f"Warning: no valid pixels for {output_path}, skip.")
        return

    dmin = np.nanmin(valid_vals)
    dmax = np.nanmax(valid_vals)
    if not np.isfinite(dmin) or not np.isfinite(dmax) or (dmax - dmin) <= 0:
        print(f"Warning: invalid depth range for {output_path}, skip.")
        return

    depth_norm = np.zeros_like(depth_np, dtype=np.float32)
    depth_norm[mask_np] = (depth_np[mask_np] - dmin) / (dmax - dmin)

    depth_u16 = np.round(depth_norm * 65535.0).clip(0, 65535).astype(np.uint16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, depth_u16)


def save_normal_map_safe(normal: torch.Tensor, mask_bool: torch.Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_normal_map(normal, mask_bool, save_path)


def save_mask_safe(mask: torch.Tensor, save_path: str, thr: float = 0.5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    m = mask
    if m.dim() == 3:
        m = m.squeeze(0)  # [H,W]
    if m.dtype.is_floating_point:
        m = (m > thr).to(torch.uint8) * 255
    else:
        m = (m > 0).to(torch.uint8) * 255
    cv2.imwrite(save_path, m.cpu().numpy())


# -------- Model --------
def build_model(args):
    return Human(
        **get_model_config(args.encoder),
        cse_config=args.cse_config,
        cse_weights=args.cse_weights,
        cse_embedder=args.cse_embedder,
    )


# -------- Inference --------
@torch.inference_mode()
def run(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = build_model(args).to(device).eval()
    load_model_checkpoint(model, args.ckpt)

    input_root = Path(args.input_path)
    output_root = Path(args.output_path)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    images = [Path(r) / f for r, _, fs in os.walk(str(input_root)) for f in fs if f.lower().endswith(exts)]
    images.sort()
    if not images:
        print(f"[Info] no images under {input_root}")
        return

    for img_path in tqdm(images, desc="Inferring", unit="img"):
        try:
            rgb_t, meta = preprocess_smart(img_path, args.img_size, args.size_divisor)  # [3,S,S], 0-1
            rgb_t = rgb_t.unsqueeze(0).to(device)                                       # [1,3,S,S]
            pred = model(rgb_t)

            depth_pred  = unpad_and_restore(pred["depth"].unsqueeze(1), meta, mode="bilinear")[0]  # [1,H,W]
            normal_pred = unpad_and_restore(pred["normal"],            meta, mode="bilinear")[0]   # [3,H,W]
            mask_pred   = unpad_and_restore(pred["mask"].unsqueeze(1), meta, mode="bilinear")[0]   # [1,H,W]

            rel  = img_path.relative_to(input_root)
            stem = rel.stem

            if args.save_depth:
                depth_mask = (mask_pred > args.mask_thr).bool() if not args.ignore_mask else torch.ones_like(mask_pred, dtype=torch.bool)
                depth_dir = output_root / "depth" / rel.parent
                save_depth_masked(depth_pred, depth_mask, str(depth_dir / f"{stem}.png"))
            if args.save_normal:
                vis_mask = (mask_pred > args.mask_thr).bool() if not args.ignore_mask else torch.ones_like(mask_pred, dtype=torch.bool)
                normal_dir = output_root / "normal" / rel.parent
                save_normal_map_safe(normal_pred, vis_mask, str(normal_dir / f"{stem}.png"))
            if args.save_mask:
                mask_dir = output_root / "mask" / rel.parent
                save_mask_safe(mask_pred, str(mask_dir / f"{stem}.png"), thr=args.mask_thr)

        except Exception as e:
            print(f"[Error] {img_path}: {e}")


# -------- CLI --------
def parse_args():
    p = argparse.ArgumentParser("Human model batch inference (0-1 input, smart padding)")
    p.add_argument("--input_path", required=True, help="Root directory of images (can include subdirectories)")
    p.add_argument("--output_path", required=True, help="Output root directory")
    p.add_argument("--ckpt", required=True, help="Path to your trained checkpoint (required)")
    p.add_argument("--encoder", type=str, default="vitl", choices=["vitb", "vitl"])
    p.add_argument("--cse-config", type=str, default=None, help='DensePose CSE config yaml')
    p.add_argument("--cse-weights", type=str, default=None, help='DensePose CSE model weights')
    p.add_argument("--cse-embedder", type=str, default=None, help='DensePose CSE embedder weights')
    p.add_argument("--img-size", type=int, default=2048, help="If set, do upper-bound scaling + replicate padding to [S,S]. "
                        "If not set, auto: scale long side to nearest multiple of size_divisor, then pad short side to square.")
    p.add_argument("--size-divisor", type=int, default=16,
                   help="Divisibility constraint for the long side in auto mode (default: 16).")
    p.add_argument("--device", type=str, default='cuda', help="cuda or cpu; auto select by default")
    # Saving options
    p.add_argument("--save-depth",  action="store_true", help="Save depth (16-bit mm PNG)")
    p.add_argument("--save-normal", action="store_true", help="Save normal map")
    p.add_argument("--save-mask",   action="store_true", help="Save mask map")
    p.add_argument("--mask-thr", type=float, default=0.5, help="Mask threshold, used for saving and normal visualization")
    p.add_argument("--ignore_mask", action="store_true", help="Do not use mask for normal visualization")
    args = p.parse_args()
    if not (args.save_depth or args.save_normal or args.save_mask):
        args.save_depth = True
        args.save_normal = True
        args.save_mask = True
    return args


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
