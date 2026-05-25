
import cv2
import bisect
import torch
import random
from typing import Dict, List, Tuple
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path

from .utils import _parse_line, _natural_key, _read_frame
from .transform import Resize_Pad, PrepareForNet, NormalizeDepthSequence, NormalizeNormal, VideoConsistentJitter

class HumanVideoDataset(Dataset):
    def __init__(
        self,
        filelist_path: str,
        data_root: str,
        size: tuple = (768, 768),
        clip_length: int = 32,
        stride: int = 1,
        mode: str = "train",
        jitter: bool = True,
        pad_to_square: bool = True,
        sampling: str = "random"
    ):
        assert mode in ("train", "val")
        self.mode = mode
        self.data_root = Path(data_root)
        self.T = clip_length
        self.stride = stride
        self.sampling = sampling

        with open(filelist_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        seq2frames = {}
        for l in lines:
            seq_id, frame_stub = _parse_line(l)
            seq2frames.setdefault(seq_id, []).append(frame_stub)
        for k in seq2frames.keys():
            seq2frames[k].sort(key=_natural_key)
        self.seq2frames = seq2frames

        self.seq_ids = list(self.seq2frames.keys())

        self.windows_per_seq = []
        for sid in self.seq_ids:
            n = len(self.seq2frames[sid])
            if self.sampling == "sliding":
                w = max(0, (n - self.T) // self.stride + 1)
            else:
                w = max(0, n - self.T + 1)
            self.windows_per_seq.append(w)

        self.windows_per_seq = np.asarray(self.windows_per_seq, dtype=np.int64)
        self.cum_windows = np.cumsum(self.windows_per_seq)
        self.total_windows = int(self.cum_windows[-1]) if len(self.cum_windows) > 0 else 0

        h, w = size
        self.ensure_multiple_of = 16
        self.resize = Resize_Pad(height=h, width=w,
                                 resize_target=True,
                                 keep_aspect_ratio=True,
                                 ensure_multiple_of=self.ensure_multiple_of,
                                 resize_method="upper_bound",
                                 image_interpolation_method=cv2.INTER_CUBIC,
                                 pad_to_square=pad_to_square)
        self.jitter = VideoConsistentJitter() if (mode=="train" and jitter) else None
        self.normalize_depth = NormalizeDepthSequence()
        self.norm_normal = NormalizeNormal()
        self.to_net = PrepareForNet()

    def __len__(self) -> int:
        if self.sampling == "sliding":
            return self.total_windows
        
        return max(1, self.total_windows)
    
    def _map_index_to_seq_start_sliding(self, idx: int) -> Tuple[str, int]:
        seq_pos = bisect.bisect_right(self.cum_windows, idx)
        seq_id = self.seq_ids[seq_pos]
        seq_start_cum = 0 if seq_pos == 0 else self.cum_windows[seq_pos - 1]
        local_idx = idx - seq_start_cum
        start = local_idx * self.stride
        return seq_id, start
    
    def _sample_seq_start_random(self) -> Tuple[str, int]:
        if self.total_windows == 0:
            valid = [sid for sid in self.seq_ids if len(self.seq2frames[sid]) >= self.T]
            if not valid:
                raise RuntimeError("No sequence has enough frames for a clip.")
            seq_id = random.choice(valid)
            n = len(self.seq2frames[seq_id])
            start = random.randint(0, n - self.T)
            return seq_id, start

        probs = self.windows_per_seq / self.windows_per_seq.sum()
        seq_idx = int(np.random.choice(len(self.seq_ids), p=probs))
        seq_id = self.seq_ids[seq_idx]
        n = len(self.seq2frames[seq_id])
        max_s = n - self.T
        start = random.randint(0, max_s)
        return seq_id, start

    def _fetch_clip(self, seq_id: str, start: int) -> List[Dict[str, np.ndarray]]:
        frames = self.seq2frames[seq_id]
        if start < 0:
            max_s = len(frames) - self.T
            start = random.randint(0, max_s)
        clip = [ _read_frame(self.data_root, seq_id, frames[start+i]) for i in range(self.T) ]
        return clip

    def __getitem__(self, idx: int):
        if self.sampling == "sliding":
            if idx < 0 or idx >= self.total_windows:
                raise IndexError(f"Index {idx} out of range [0, {self.total_windows}).")
            seq_id, s = self._map_index_to_seq_start_sliding(idx)
            clip = self._fetch_clip(seq_id, s)
        else:
            seq_id, s = self._sample_seq_start_random()
            clip = self._fetch_clip(seq_id, s)

        clip = [self.resize(smp) for smp in clip]

        if self.jitter is not None:
            clip = self.jitter(clip)

        out = []
        clip = [self.norm_normal(smp) if "normal" in smp else smp for smp in clip]
        clip = self.normalize_depth(clip)
        out = [self.to_net(smp) for smp in clip]

        imgs  = torch.stack([torch.from_numpy(s["image"]) for s in out], dim=0)         # [T,3,H,W]
        depth = torch.stack([torch.from_numpy(s["depth"]) for s in out], dim=0)         # [T,H,W]
        depth_rel = torch.stack([torch.from_numpy(s["depth_relative"]) for s in out],0) # [T,H,W]
        normal= torch.stack([torch.from_numpy(s["normal"]) for s in out], dim=0) if "normal" in out[0] else None
        mask  = torch.stack([torch.from_numpy(s["mask"]).bool() for s in out], dim=0)

        valid = (depth > 0.001) & mask
        sample = {
            "image": imgs,                # [T,3,H,W]
            "normal": normal,             # [T,3,H,W]
            "depth": depth,               # [T,H,W]
            "depth_relative": depth_rel,  # [T,H,W]
            "mask": mask,                 # [T,H,W] bool
            "valid_mask": valid,          # [T,H,W] bool
        }
        return sample
