import os, cv2, argparse, gc
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from model.human_temporal import Human as HumanVideo
from vis.utils import save_normal_map
from utils.dinov3_helper import get_model_config, load_model_checkpoint

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

# ---------- I/O ----------
def list_frames(dir_path: Path):
    return sorted([f for f in dir_path.iterdir() if f.suffix.lower() in IMG_EXTS])

# ---------- resize / pad helpers (square upper-bound) ----------
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


# ---------- small utils ----------
def gen_keyframes(window_len: int, overlap: int, interp_len: int):
    assert 0 < overlap < window_len
    assert 0 < interp_len <= overlap
    align_len = overlap - interp_len
    if align_len <= 0:
        return list(range(window_len - interp_len, window_len))
    span = window_len - interp_len
    if align_len == 1:
        anchors = [0]
    else:
        step = span / align_len
        anchors = [int(round(i*step)) for i in range(align_len)]
        anchors = [min(a, span-1) for a in anchors]
        anchors[0] = 0
    tail = list(range(window_len - interp_len, window_len))
    return anchors + tail

def lerp(a, b, t):
    return (1.0 - t) * a + t * b

def interp_block(pre_list, post_list, K):
    out = []
    for i in range(K):
        t = (i+1) / (K+1)
        out.append(lerp(pre_list[i], post_list[i], t))
    return out

def slerp_vec(a: np.ndarray, b: np.ndarray, t: float):
    dot = np.sum(a*b, axis=-1, keepdims=True)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    b = b * sign
    dot = np.sum(a*b, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    den = np.sin(omega)
    w0 = np.sin((1.0 - t) * omega) / (den + 1e-8)
    w1 = np.sin(t * omega) / (den + 1e-8)
    out = w0 * a + w1 * b
    out /= (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)
    return out

def get_interpolate_normals(pre_list, post_list):
    assert len(pre_list) == len(post_list)
    K = len(pre_list)
    if K == 0:
        return []
    if K == 1:
        a, b = pre_list[0], post_list[0]
        if a.ndim == 3 and a.shape[0] == 3:
            a = a.transpose(1,2,0); b = b.transpose(1,2,0)
            out = slerp_vec(a, b, 0.5).transpose(2,0,1)
        else:
            out = slerp_vec(a, b, 0.5)
        return [out]
    ts = np.linspace(0.0, 1.0, K).tolist()
    outs = []
    for i in range(K):
        a, b = pre_list[i], post_list[i]
        if a.ndim == 3 and a.shape[0] == 3:
            a = a.transpose(1,2,0); b = b.transpose(1,2,0)
            o = slerp_vec(a, b, ts[i]).transpose(2,0,1)
        else:
            o = slerp_vec(a, b, ts[i])
        outs.append(o)
    return outs


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1

def get_interpolate_frames(frame_list_pre, frame_list_post):
    assert len(frame_list_pre) == len(frame_list_post)
    min_w = 0.0
    max_w = 1.0
    step = (max_w - min_w) / (len(frame_list_pre)-1)
    post_w_list = [min_w] + [i * step for i in range(1,len(frame_list_pre)-1)] + [max_w]
    interpolated_frames = []
    for i in range(len(frame_list_pre)):
        interpolated_frames.append(frame_list_pre[i] * (1-post_w_list[i]) + frame_list_post[i] * post_w_list[i])
    return interpolated_frames

# ---------- per-sequence pipeline (predicted masks only) ----------
def process_one_sequence(seq_dir: Path,
                         out_depth_root: Path, out_normal_root: Path, out_mask_root: Path,
                         args, model, window_len, overlap, interp_len):
    frame_paths = list_frames(seq_dir)
    if len(frame_paths) == 0:
        return

    frames_tensor, frame_meta = [], []
    for fp in frame_paths:
        it, meta_i = preprocess_smart(fp, args.img_size)
        frames_tensor.append(it); frame_meta.append(meta_i)
    frames_tensor = torch.stack(frames_tensor, 0)  # [N,3,S,S]
    N = frames_tensor.shape[0]

    step = window_len - overlap
    append_len = (step - (N % step)) % step + (window_len - step)
    frames_pad = torch.cat([frames_tensor, frames_tensor[-1:].repeat(append_len,1,1,1)], 0)
    total = frames_pad.shape[0]

    keyframes = gen_keyframes(window_len, overlap, interp_len) if not args.keyframes else args.keyframes
    assert len(keyframes) == overlap, f"len(keyframes) must equal overlap, got {len(keyframes)} vs {overlap}"

    depth_chunks, normal_chunks, mask_chunks = [], [], []
    prev_input = None
    for s in tqdm(range(0, total - window_len + 1, step), desc=f"Infer[{seq_dir.name}]", unit="win"):
        cur = frames_pad[s:s+window_len].to(args.device).unsqueeze(0)  # [1,T,3,S,S]
        if prev_input is not None:
            cur[:, :overlap, ...] = prev_input[:, keyframes, ...]

        out = model(cur)

        d_win, n_win, m_win = [], [], []
        for t in range(window_len):
            mid = min(s+t, N-1)
            d = unpad_and_restore(out["depth"][t].unsqueeze(0), frame_meta[mid], mode="bilinear")[0].cpu().numpy().astype(np.float32)
            n = unpad_and_restore(out["normal"][t], frame_meta[mid], mode="bilinear").cpu().numpy()
            m = unpad_and_restore(out["mask"][t].unsqueeze(0), frame_meta[mid])[0]
            m = m.clamp(0,1).cpu().numpy().astype(np.float32)
            d_win.append(d)
            n_win.append(n) 
            m_win.append(m)

        depth_chunks.append(d_win)
        normal_chunks.append(n_win)
        mask_chunks.append(m_win)
        prev_input = cur.detach()

    del frames_pad
    gc.collect()

    depth_list = [frm.astype(np.float32) for win in depth_chunks for frm in win]
    mask_list  = [msk.astype(np.float32) for win in mask_chunks  for msk in win]

    # ---- depth stitching: window-wise affine on foreground + tail interpolation ----
    depth_aligned = []
    ref_align = []
    ref_align_mask = []
    align_len = overlap - interp_len
    kf_align_list = keyframes[:align_len] if align_len > 0 else []

    for base in range(0, len(depth_list), window_len):
        if len(depth_aligned) == 0:
            depth_aligned += depth_list[:window_len]
            for kf in kf_align_list:
                ref_align.append(depth_list[base + kf])
                ref_align_mask.append(mask_list[base + kf])
        else:
            curr_align = [depth_list[base + i] for i in range(len(kf_align_list))] if align_len > 0 else []
            curr_mask  = [mask_list[base + i]  for i in range(len(kf_align_list))] if align_len > 0 else []

            if len(curr_align) == len(ref_align) and len(curr_align) > 0:
                bin_pairs = []
                for cm, rm in zip(curr_mask, ref_align_mask):
                    cm_bin = (cm >= args.pred_mask_thr).astype(np.float32)
                    rm_bin = (rm >= args.pred_mask_thr).astype(np.float32)
                    bin_pairs.append((cm_bin, rm_bin))

                pred_flat = np.concatenate([c.reshape(-1) for c in curr_align], axis=0)
                tgt_flat  = np.concatenate([r.reshape(-1) for r in ref_align],  axis=0)
                msk_flat  = np.concatenate([(cm_bin * rm_bin).reshape(-1) for cm_bin, rm_bin in bin_pairs], axis=0).astype(np.float32)

                if np.any(msk_flat > 0):
                    scale, shift = compute_scale_and_shift(pred_flat, tgt_flat, msk_flat)
                else:
                    scale, shift = 1.0, 0.0
            else:
                scale, shift = 1.0, 0.0


            pre_tail  = depth_aligned[-interp_len:]
            post_tail = [(depth_list[base + i] * scale + shift).astype(np.float32) for i in range(align_len, overlap)]
            if len(pre_tail)==len(post_tail) and len(pre_tail)>0:
                depth_aligned[-interp_len:] = get_interpolate_frames(pre_tail, post_tail)
            else:
                depth_aligned[-interp_len:] = interp_block(pre_tail, post_tail, interp_len)

            for i in range(overlap, window_len):
                new_depth = (depth_list[base + i] * scale + shift).astype(np.float32)
                depth_aligned.append(new_depth)

            if align_len > 0:
                ref_align = ref_align[:1]
                ref_align_mask = ref_align_mask[:1]
                for kf in kf_align_list[1:]:
                    new_ref = (depth_list[base + kf] * scale + shift).astype(np.float32)
                    ref_align.append(new_ref)
                    ref_align_mask.append(mask_list[base + kf])

    depth_aligned = depth_aligned[:len(frame_paths)]

    # ---- normal & mask stitching: tail interpolation only ----
    normal_aligned, mask_aligned = [], []
    for w in range(len(normal_chunks)):
        n_win = normal_chunks[w]
        m_win = mask_chunks[w]
        if w == 0:
            normal_aligned.extend(n_win)
            mask_aligned.extend(m_win)
        else:
            pre_n  = normal_aligned[-interp_len:]
            post_n = n_win[align_len:overlap]
            
            if len(pre_n) == len(post_n) and len(pre_n) > 0:
                normal_aligned[-interp_len:] = get_interpolate_normals(pre_n, post_n)
            else:
                K = min(len(pre_n), len(post_n), interp_len)
                if K > 0:
                    normal_aligned[-K:] = get_interpolate_normals(pre_n[-K:], post_n[:K])

            for j in range(overlap, len(n_win)):
                nxt = n_win[j]
                if nxt.ndim == 3 and nxt.shape[0] == 3:
                    nxt = nxt / (np.linalg.norm(nxt, axis=0, keepdims=True) + 1e-8)
                else:
                    nxt = nxt / (np.linalg.norm(nxt, axis=2, keepdims=True) + 1e-8)
                normal_aligned.append(nxt)
                mask_aligned.append(m_win[j])

    normal_aligned = normal_aligned[:len(frame_paths)]
    mask_aligned   = mask_aligned[:len(frame_paths)]

    # ---------- outputs ----------
    seq_depth_dir  = out_depth_root  / seq_dir.name
    seq_normal_dir = out_normal_root / seq_dir.name
    seq_mask_dir   = out_mask_root   / seq_dir.name
    seq_depth_dir.mkdir(parents=True, exist_ok=True)
    seq_normal_dir.mkdir(parents=True, exist_ok=True)
    seq_mask_dir.mkdir(parents=True, exist_ok=True)

    # global foreground min/max from predicted masks
    bin_masks, fg_vals = [], []
    for i in range(len(mask_aligned)):
        m_bin = (mask_aligned[i] >= args.pred_mask_thr)
        bin_masks.append(m_bin)
        if m_bin.any():
            fg_vals.append(depth_aligned[i][m_bin])

    if len(fg_vals) > 0:
        fg_cat = np.concatenate(fg_vals, axis=0)
        fg_min = float(np.percentile(fg_cat, 2.0))
        fg_max = float(np.percentile(fg_cat, 98.0))
    else:
        vals = np.concatenate([d.reshape(-1) for d in depth_aligned], axis=0)
        fg_min = float(np.percentile(vals, 2.0))
        fg_max = float(np.percentile(vals, 98.0))

    if fg_max <= fg_min:
        fg_max = fg_min + 1e-6

    for i, fp in enumerate(tqdm(frame_paths, desc=f"Save[{seq_dir.name}]", unit="frm")):
        stem = Path(fp).stem

        d = depth_aligned[i].astype(np.float32)
        n = normal_aligned[i]
        m = bin_masks[i]

        dn = (d - fg_min) / (fg_max - fg_min)
        u16 = np.clip(np.round(dn * 65535.0), 0, 65535).astype(np.uint16)
        u16[~m] = 0
        cv2.imwrite(str(seq_depth_dir / f"{stem}.png"), u16)

        m_t = torch.from_numpy(m)

        n_t = torch.from_numpy(n).float()
        if n_t.ndim == 3 and n_t.shape[-1] == 3:
            n_t = n_t.permute(2, 0, 1).contiguous()

        save_normal_map(n_t, m_t.unsqueeze(0), str(seq_normal_dir / f"{stem}.png"))

        pm_bin = (m.astype(np.uint8)) * 255
        cv2.imwrite(str(seq_mask_dir / f"{stem}.png"), pm_bin)

# ---------- main ----------
@torch.inference_mode()
def run(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True
    args.device = device

    window_len = args.frames
    overlap = args.overlap
    interp_len = args.interp_len
    assert 0 < overlap < window_len
    assert 0 < interp_len <= overlap

    model = HumanVideo(
        **get_model_config(args.encoder, num_frames=window_len),
        cse_config=args.cse_config,
        cse_weights=args.cse_weights,
        cse_embedder=args.cse_embedder,
    ).to(device).eval()
    load_model_checkpoint(model, args.ckpt)

    in_path = Path(args.input_path).resolve()
    assert in_path.exists(), f"input_path not found: {in_path}"

    def has_frames(p: Path):
        return len(list_frames(p)) > 0

    if in_path.is_dir() and has_frames(in_path):
        seq_dirs = [in_path]
    elif in_path.is_dir():
        seq_dirs = [d for d in in_path.iterdir() if d.is_dir()]
        assert len(seq_dirs)>0, f"no sequence subfolders under {in_path}"
    else:
        raise AssertionError(f"input_path must be a directory: {in_path}")

    out_depth_root  = Path(args.output_path) / "depth"
    out_normal_root = Path(args.output_path) / "normal"
    out_mask_root   = Path(args.output_path) / "mask"
    out_depth_root.mkdir(parents=True, exist_ok=True)
    out_normal_root.mkdir(parents=True, exist_ok=True)
    out_mask_root.mkdir(parents=True, exist_ok=True)

    for seq_dir in seq_dirs:
        process_one_sequence(seq_dir,
                             out_depth_root, out_normal_root, out_mask_root,
                             args, model, window_len, overlap, interp_len)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("Sliding-window video depth/normal/mask; window-wise affine on depth using predicted masks; tail interpolation")
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder", default="vitl", choices=["vitb", "vitl"])
    p.add_argument("--cse-config", type=str, default=None, help='DensePose CSE config yaml')
    p.add_argument("--cse-weights", type=str, default=None, help='DensePose CSE model weights')
    p.add_argument("--cse-embedder", type=str, default=None, help='DensePose CSE embedder weights')
    p.add_argument("--img_size", type=int, default=1152)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--overlap", type=int, default=10)
    p.add_argument("--interp_len", type=int, default=8)
    p.add_argument("--pred_mask_thr", type=float, default=0.5)
    p.add_argument("--keyframes", type=str, default="")
    args = p.parse_args()
    args.keyframes = [int(x) for x in args.keyframes.split(",") if x.strip()!=""] if args.keyframes else []
    return args

if __name__ == "__main__":
    args = parse_args()
    run(args)
