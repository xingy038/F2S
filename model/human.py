import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .dinov3.models.vision_transformer import DINOv3
from .block.dpt import DPT, Head
from .block.cse_pos import CSEPosEncoder

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CSE_CONFIG = _ROOT / "configs" / "densepose" / "densepose_rcnn_R_50_FPN_DL_s1x.yaml"
_DEFAULT_CSE_WEIGHTS = _ROOT / "checkpoints" / "model_final_e96218.pkl"
_DEFAULT_CSE_EMBEDDER = _ROOT / "checkpoints" / "phi_smpl_27554_256.pkl"

class Human(nn.Module):
    def __init__(
        self, 
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        cse_config: str | None = None,
        cse_weights: str | None = None,
        cse_embedder: str | None = None,
    ):
        super().__init__()
        if encoder not in {"vitb", "vitl"}:
            raise ValueError("The open-source release only supports encoder='vitb' or 'vitl'.")

        self.intermediate_layer_idx = {
            "vitb": [2, 5, 8, 11],
            "vitl": [5, 11, 17, 23],
        }
        self.encoder = encoder
        self.patch_size = 16
        self.pretrained = DINOv3(model_name=encoder)
        self.cse_config = Path(cse_config) if cse_config else _DEFAULT_CSE_CONFIG
        self.cse_weights = Path(cse_weights) if cse_weights else _DEFAULT_CSE_WEIGHTS
        self.cse_embedder = Path(cse_embedder) if cse_embedder else _DEFAULT_CSE_EMBEDDER

        self.cse = None

        self.dpt = DPT(self.pretrained.embed_dim, features, out_channels=out_channels, use_clstoken=use_clstoken)
        self.depth_head = Head(features, 1)
        self.mask_head = Head(features, 1)
        self.normal_head = Head(features, 3)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)
    
    def init_cse(self, device):
        if self.cse is None:
            self.cse = CSEPosEncoder(
                cfg_path=str(self.cse_config),
                weights_path=str(self.cse_weights),
                embedder_path=str(self.cse_embedder),
                device=device,
                tau=0.5,
            )
    
    def forward(self, x):
        self.init_cse(x.device)
        pos = self.cse(x)

        x = (x - self._resnet_mean) / self._resnet_std
        ori_h, ori_w = x.shape[-2], x.shape[-1]
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        features = self.dpt(features, x, patch_h, patch_w, pos)

        depth = self.depth_head(features, ori_h, ori_w)
        mask = self.mask_head(features, ori_h, ori_w)
        normal = self.normal_head(features, ori_h, ori_w)

        normal = F.normalize(normal, dim=1)
        mask = mask.squeeze(1).sigmoid()

        return {
            "depth": depth.squeeze(1), 
            "normal": normal,
            "mask": mask,
        }
