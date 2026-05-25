import os
import fnmatch
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim.swa_utils
import torch.backends.cudnn as cudnn

def schedule_weight(it: int, start_it: int, end_it: int, w0: float, w1: float) -> float:
    """
    Cosine-eased schedule for a scalar weight.
    - it: current iteration (0-based)
    - start_it/end_it: ramp window [start_it, end_it]
    - w0 -> w1: start/end values
    """
    if it <= start_it:
        return w0
    if it >= end_it:
        return w1
    t = (it - start_it) / float(end_it - start_it)
    import math
    t = 0.5 - 0.5 * math.cos(math.pi * t)  # smooth rise
    return w0 + (w1 - w0) * t

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.enabled = True
    cudnn.benchmark = True

def infinite_loader(data_loader):
    """Infinite iterator with proper set_epoch for DistributedSampler."""
    epoch = 0
    while True:
        if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)
        for batch in data_loader:
            yield batch
        epoch += 1

def any_match(s: str, patterns: list[str]) -> bool:
    """Checks if a string matches any of the given patterns."""
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)

def build_optimizer(model: nn.Module, optimizer_config: dict) -> torch.optim.Optimizer:
    """Builds an optimizer from a configuration dictionary."""
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    groups = []
    for grp_cfg in optimizer_config['params']:
        inc = grp_cfg['params']['include']
        exc = grp_cfg['params'].get('exclude', [])
        # find matching names
        names = [n for n in named if any_match(n, inc) and not any_match(n, exc)]
        assert names, f"no params match include={inc} exclude={exc}"
        params = [named.pop(n) for n in names]
        opts = {k: v for k, v in grp_cfg.items() if k != 'params'}
        groups.append({'params': params, **opts})
    # ensure none left out
    assert not named, f"these params require grad but were not included: {list(named)}"
    Optimizer = getattr(torch.optim, optimizer_config['type'])
    return Optimizer(groups)

def _is_cse_key(k: str) -> bool:
    if k.startswith("module."):
        k = k[len("module."):]
    return k.startswith("cse.") or ".cse." in k

def _drop_cse(sd: dict) -> dict:
    return {k: v for k, v in sd.items() if not _is_cse_key(k)}

def save_checkpoint(accelerator, model, iter_count, args, logger):
    if not accelerator.is_main_process:
        return
    ckpt_dir = os.path.join(args.save_path, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"iter_{iter_count:06d}.pt")

    state_dict = accelerator.get_state_dict(model)

    state_dict = _drop_cse(state_dict)

    payload = {
        "model": state_dict,
    }
    torch.save(payload, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")
