from __future__ import annotations

from pathlib import Path

import torch

SUPPORTED_ENCODERS = ("vitb", "vitl")

MODEL_CONFIGS = {
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def get_model_config(encoder: str, *, num_frames: int | None = None) -> dict:
    encoder = encoder.lower()
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported encoder '{encoder}'. Supported: {SUPPORTED_ENCODERS}")
    cfg = dict(MODEL_CONFIGS[encoder])
    if num_frames is not None:
        cfg["num_frames"] = num_frames
    return cfg


def _is_tensor_dict(obj) -> bool:
    return isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values())


def _unwrap_state_dict(obj):
    if _is_tensor_dict(obj):
        return obj
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint is not a dict-like object.")

    preferred_keys = (
        "model",
        "state_dict",
        "model_state_dict",
        "teacher",
        "student",
        "network",
    )
    for key in preferred_keys:
        value = obj.get(key)
        if isinstance(value, dict):
            try:
                return _unwrap_state_dict(value)
            except ValueError:
                pass

    for value in obj.values():
        if isinstance(value, dict):
            try:
                return _unwrap_state_dict(value)
            except ValueError:
                continue

    raise ValueError("Could not find a tensor state_dict inside checkpoint.")


def _strip_prefixes(key: str) -> str:
    prefixes = ("module.", "backbone.", "teacher.", "student.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _map_backbone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state_dict.items():
        key = _strip_prefixes(key)
        if not key.startswith("pretrained."):
            key = f"pretrained.{key}"
        mapped[key] = value
    return mapped


def load_pretrained_backbone(model: torch.nn.Module, checkpoint_path: str, logger=None):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {path}")

    raw = torch.load(path, map_location="cpu")
    state_dict = _unwrap_state_dict(raw)
    mapped = _map_backbone_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(mapped, strict=False)

    if logger is not None:
        logger.info(
            f"Loaded DINOv3 backbone from {path}. "
            f"Missing={len(missing)}, Unexpected={len(unexpected)}"
        )
    return missing, unexpected


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, logger=None):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    raw = torch.load(path, map_location="cpu")
    state_dict = _unwrap_state_dict(raw)
    cleaned = {_strip_prefixes(k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if logger is not None:
        logger.info(
            f"Loaded model checkpoint from {path}. "
            f"Missing={len(missing)}, Unexpected={len(unexpected)}"
        )
    return missing, unexpected
