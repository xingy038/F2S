from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import cv2

from .transform import Resize_Pad, PrepareForNet, Jittering, Blurring, ShotNoise, JPEGCompression, NormalizeDepth
from .utils import read_SynthHuman_raw_sample

class HumanDataset_SynthHuman(Dataset):
    def __init__(
        self,
        filelist_path: str,
        data_path: str,
        mode: str,
        size: tuple = (768, 768),
    ):
        if mode not in ("train"):
            raise NotImplementedError("Only 'train' and 'val' supported.")
        self.mode = mode
        self.data_path = Path(data_path)
        h, w = size

        with open(filelist_path, "r") as f:
            self.filelist = [
                Path(line.strip().split()[0])
                for line in f
                if line.strip()
            ]

        base_transforms = [
            Resize_Pad(
                    height=h, width=w,
                    resize_target=(mode == "train"),
                    keep_aspect_ratio=True,
                    ensure_multiple_of=16,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
            ]
        if self.mode == "train":
            extras = [
                Jittering(),
                ShotNoise(shot_noise_prob=0.5, k_range=(100,10000), sigma_range=(0.0, 0.02)),
                JPEGCompression(prob=0.3, quality_range=(50, 100)),
                Blurring(blur_prob=0.2, kernel_size_range=(3,5)),
                NormalizeDepth(),
                PrepareForNet(),
            ]
        self.transform = Compose(base_transforms + extras)

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, idx: int):
        rel_path = self.filelist[idx]
        raw = read_SynthHuman_raw_sample(self.data_path, rel_path)
        sample = self.transform(raw)

        return self._to_tensor(sample)

    def _to_tensor(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        if self.mode == "train":
            image = torch.from_numpy(data["image"])
            depth = torch.from_numpy(data["depth"])
            depth_relative = torch.from_numpy(data["depth_relative"])
            normal = torch.from_numpy(data["normal"])
            mask = torch.from_numpy(data["mask"]).bool()
            valid = (depth > 0.001) & mask

            return {
                "image": image,
                "depth": depth,
                "depth_relative": depth_relative,
                "normal": normal,
                "mask": mask,
                "valid_mask": valid,
            }
