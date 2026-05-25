import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from densepose import add_densepose_config

class CSEPosEncoder(nn.Module):
    def __init__(
        self,
        cfg_path: str,
        weights_path: str,
        embedder_path: str,
        device: torch.device,
        tau: float = 0.5,
    ):
        super().__init__()
        cfg_path = Path(cfg_path).expanduser().resolve()
        weights_path = Path(weights_path).expanduser().resolve()
        embedder_path = Path(embedder_path).expanduser().resolve()

        for path in (cfg_path, weights_path, embedder_path):
            if not path.exists():
                raise FileNotFoundError(f"Required CSE asset not found: {path}")

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(str(cfg_path))
        cfg.MODEL.WEIGHTS = str(weights_path)
        cfg.MODEL.DEVICE = str(device)

        embedders = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS
        embedder_names = list(embedders.keys())
        if len(embedder_names) != 1:
            raise ValueError(f"Expected exactly one CSE embedder, got {embedder_names}")
        embedders[embedder_names[0]].INIT_FILE = str(embedder_path)

        self.cfg = cfg

        self.model = build_model(cfg).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        checkpointer = DetectionCheckpointer(self.model)
        state = checkpointer._load_file(cfg.MODEL.WEIGHTS)
        model_state = state.get("model", {})
        for k in ("pixel_mean", "pixel_std"):
            if k in model_state:
                model_state.pop(k)
        checkpointer._load_model(state)

        self.resize_aug = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

        self.tau = tau
        self._grid_cache = {}  # (H,W) -> grid
        self._mesh_cache = {}  # (H,W) -> (yy, xx)

        self.device = device

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    def _get_grid(self, H: int, W: int):
        key = (H, W)
        if key in self._grid_cache:
            return self._grid_cache[key]
        yy = torch.linspace(0, H - 1, H, device=self.device)
        xx = torch.linspace(0, W - 1, W, device=self.device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        self._grid_cache[key] = (yy, xx)
        return self._grid_cache[key]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H0, W0 = x.shape
        yy, xx = self._get_grid(H0, W0)

        x_np = (x.clamp(0, 1) * 255.0).byte().permute(0, 2, 3, 1).contiguous()  # [B,H,W,3], uint8, RGB
        x_np = x_np[..., [2,1,0]].cpu().numpy()  # BGR

        images = []
        for b in range(B):
            img_bgr = x_np[b]
            tfm = self.resize_aug.get_transform(img_bgr)
            img_bgr_resized = tfm.apply_image(img_bgr)
            img_chw = torch.tensor(img_bgr_resized.transpose(2,0,1).copy(), dtype=torch.float32)
            images.append({"image": img_chw, "height": H0, "width": W0})

        outputs = self.model(images)

        out_list = []
        eps = 1e-6

        for b in range(B):
            inst = outputs[b]["instances"].to(self.device)
            if len(inst) == 0 or not hasattr(inst, "pred_densepose"):
                out_list.append(torch.zeros(16, H0, W0, device=self.device))
                continue

            boxes = inst.pred_boxes.tensor.float()
            scores = inst.scores.float()
            dp = inst.pred_densepose

            w = (scores / max(eps, self.tau)).softmax(dim=0)  # [N]

            num = torch.zeros(16, H0, W0, device=self.device)
            den = torch.zeros(1,  H0, W0, device=self.device)

            for i in range(len(dp)):
                E = torch.as_tensor(dp[i].embedding[0], device=self.device, dtype=torch.float32)  # [16,h,w]
                Sfg_logits = torch.as_tensor(dp[i].coarse_segm[0,1], device=self.device, dtype=torch.float32)
                Sfg = Sfg_logits.sigmoid().view(1, *Sfg_logits.shape[-2:])  # [1,h,w]

                x1, y1, x2, y2 = boxes[i]
                gx = (xx - x1) / torch.clamp(x2 - x1, min=1.0) * 2 - 1
                gy = (yy - y1) / torch.clamp(y2 - y1, min=1.0) * 2 - 1
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1,H,W,2]

                E_full = F.grid_sample(E.unsqueeze(0), grid, mode="bilinear", align_corners=True).squeeze(0)  # [16,H,W]
                M_full = F.grid_sample(Sfg.unsqueeze(0),grid, mode="bilinear", align_corners=True).squeeze(0)  # [1,H,W]

                wi = w[i].view(1,1,1)
                num = num + E_full * (M_full * wi)
                den = den + (M_full * wi)

            P = num / (den + eps)
            out_list.append(P)

        return torch.stack(out_list, 0)  # [B,16,H,W]
