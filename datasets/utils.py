from typing import *

import OpenEXR
import Imath
import re
import numpy as np
from pathlib import Path
import json
import os
from PIL import Image

_cam_cache: Dict[Path, Dict[int, Dict[str, float]]] = {}

def read_exr(path, channels='V'):
    f = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if isinstance(channels, str):
        raw = f.channel(channels, pt)
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.reshape((H, W))

    arrs = []
    for c in channels:
        raw = f.channel(c, pt)
        a = np.frombuffer(raw, dtype=np.float32).reshape((H, W))
        arrs.append(a)
    return np.stack(arrs, axis=-1)


def extract_frame_id(file_name: str) -> int:
    m = re.search(r'(\d+)', file_name)
    if m is None:
        raise ValueError(f'Cannot parse frame id from {file_name}')
    return int(m.group(1))

def _parse_line(p: str) -> Tuple[str, str]:
    parts = Path(p.strip()).parts
    assert len(parts) >= 3 and parts[0] == "dynamic", f"Bad path: {p}"
    seq = "/".join(parts[:2+1]) if parts[0]=="dynamic" else "/".join(parts[:-1])
    seq_id = "/".join(parts[:2+1-1])
    frame_stub = parts[-1]
    return seq_id, frame_stub

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def _read_frame(data_root: Path, seq_id: str, frame_stub: str) -> Dict[str, np.ndarray]:
    base = str(data_root)
    scene, fname = frame_stub.split('_')
    img_path    = os.path.join(base, seq_id, scene, "image", f"rgba_{fname}.jpg")
    depth_path  = os.path.join(base, seq_id, scene, "depth", f"depth_{fname}.exr")
    normal_path = os.path.join(base, seq_id, scene, "normal", f"normal_{fname}.png")
    mask_path   = os.path.join(base, seq_id, scene, "mask", f"mask_{fname}.png")

    image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    depth = read_exr(depth_path).astype(np.float32)
    depth[depth > 100.0] = 0.0
    normal = np.asarray(Image.open(normal_path), dtype=np.float32)  # HxWx3, 0..255
    mask = (np.asarray(Image.open(mask_path), dtype=np.uint8) > 127).astype(np.float32)

    return {
        "image":  image,
        "depth":  depth,
        "normal": normal,
        "mask":   mask,
    }

###################################################################################

def read_raw_sample(mode: str, data_path: Path, relative_path: Path) -> Dict[str, np.ndarray]:
    """
    Read raw image, depth, normal, mask (and intrinsic if train) 
    as numpy arrays, given mode, base data_path, and a Path entry.
    """
    parts = relative_path.parts
    if mode == "train":
        root, scene, cam_file = parts[:3]
        cam_mode, fname = cam_file.rsplit("_", 1)
        cam_dir = data_path / root / scene / cam_mode
        frame_id = extract_frame_id(fname)

        # load and cache camera parameters
        if cam_dir not in _cam_cache:
            data = json.loads((cam_dir / "camera_params.json").read_text())
            _cam_cache[cam_dir] = {c["frame"]: c for c in data}
        cam = _cam_cache[cam_dir][frame_id]

        K = np.array([
            [cam["fx"], 0.0,       cam["cx"]],
            [0.0,       cam["fy"], cam["cy"]],
            [0.0,       0.0,       1.0      ],
        ], dtype=np.float32)

        image = (
            np.asarray(
                Image.open(cam_dir / "image" / f"rgba_{fname}.jpg")
                     .convert("RGB"),
                dtype=np.float32,
            )
        ) / 255.0

        depth = read_exr(str(cam_dir / "depth" / f"depth_{fname}.exr")).astype(np.float32)
        depth[depth > 100.0] = 0.0

        normal = np.asarray(
            Image.open(cam_dir / "normal" / f"normal_{fname}.png"),
            dtype=np.float32
        )
        mask = (
            (np.asarray(
                Image.open(cam_dir / "mask" / f"mask_{fname}.png"),
                dtype=np.uint8,
            ) > 127)
            .astype(np.float32)
        )

        return {
            "image":     image,
            "depth":     depth,
            "normal":    normal,
            "mask":      mask,
            "intrinsic": K,
            "mask_ori":  mask
        }
    else:
        scene, fname = parts[0].rsplit('_', 1)
        base = str(data_path)
        img_path    = os.path.join(base, "image", scene, f"{fname}.jpg")
        depth_path  = os.path.join(base, "depth",  scene, f"{fname}.exr")
        normal_path = os.path.join(base, "normal", scene, f"{fname}.png")
        mask_path   = os.path.join(base, "mask",   scene, f"{fname}.png")

        image = (
            np.asarray(
                Image.open(img_path).convert("RGB"),
                dtype=np.float32,
            )
        ) / 255.0
        depth = read_exr(depth_path).astype(np.float32)
        depth[depth > 100.0] = 0.0
        normal = np.asarray(Image.open(normal_path), dtype=np.float32)
        mask = (
            (np.asarray(
                Image.open(mask_path),
                dtype=np.uint8,
            ) > 127)
            .astype(np.float32)
        )

        return {
            "image":  image,
            "depth":  depth,
            "normal": normal,
            "mask":   mask,
        }
    
###################################################################################
    
def read_SynthHuman_raw_sample(data_path: Path, relative_path: Path) -> Dict[str, np.ndarray]:
    """
    Read raw image, depth, normal, mask (and intrinsic if train) 
    as numpy arrays, given mode, base data_path, and a Path entry.
    """

    index = Path(relative_path).stem.split("_", 1)[1]
    base_dir = data_path 

    cam_txt = base_dir / f"cam_{index}.txt"
    with cam_txt.open("r") as f:
        lines = f.read().splitlines()
    K = np.array([
        [float(x) for x in lines[0].split()],
        [float(x) for x in lines[1].split()],
        [float(x) for x in lines[2].split()],
    ], dtype=np.float32)

    img_path = base_dir / f"rgb_{index}.png"
    image = np.asarray(
        Image.open(img_path).convert("RGB"),
        dtype=np.float32
    ) / 255.0

    alpha_path = base_dir / f"alpha_{index}.png"
    alpha = (
        (np.asarray(
            Image.open(alpha_path).convert("L"),
            dtype=np.uint8
        ) > 127)
        .astype(np.float32)
    )

    depth_path = base_dir / f"depth_{index}.exr"
    depth = read_exr(str(depth_path), channels='Y').astype(np.float32)
    depth[depth == 65504] = 0.0
    depth = depth / 100.0

    normal_path = base_dir / f"normal_{index}.exr"
    normal = read_exr(str(normal_path), channels=['B','G','R']).astype(np.float32)

    return {
        "image":     image,
        "mask":      alpha,
        "depth":     depth,
        "normal":    normal,
        "intrinsic": K,
    }
###################################################################################

