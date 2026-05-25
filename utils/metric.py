import torch

def eval_depth(pred, target, mask):
    assert pred.shape == target.shape
    assert mask.shape == pred.shape

    mask = mask.bool()
    pred = pred[mask]
    target = target[mask]

    thresh = torch.max(target / pred, pred / target)

    d1 = torch.mean((thresh < 1.25).float())
    d2 = torch.mean((thresh < 1.25 ** 2).float())
    d3 = torch.mean((thresh < 1.25 ** 3).float())

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(diff ** 2 / target)

    rmse = torch.sqrt(torch.mean(diff ** 2))
    rmse_log = torch.sqrt(torch.mean(diff_log ** 2))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(diff_log.pow(2).mean() - 0.5 * diff_log.mean().pow(2))

    return {
        'd1': d1.item(), 'd2': d2.item(), 'd3': d3.item(),
        'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(),
        'rmse': rmse.item(), 'rmse_log': rmse_log.item(),
        'log10': log10.item(), 'silog': silog.item()
    }

def compute_scale_and_shift_torch(prediction, target, mask):
    a_00 = torch.sum(mask * prediction * prediction)
    a_01 = torch.sum(mask * prediction)
    a_11 = torch.sum(mask)
    b_0 = torch.sum(mask * prediction * target)
    b_1 = torch.sum(mask * target)
    det = a_00 * a_11 - a_01 * a_01
    if det > 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale = 0.0
        shift = 0.0
    return scale, shift

def eval_depth_with_scale(pred, target, mask):
    scale, shift = compute_scale_and_shift_torch(pred, target, mask)
    pred_aligned = pred * scale + shift
    metrics = eval_depth(pred_aligned, target, mask)
    return metrics

def eval_normal(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    Evaluate surface normal prediction against ground truth.
    Supports shapes:
        pred/target: [3,H,W] | [H,W,3] | [1,3,H,W] | [1,H,W,3]
        mask:        [H,W] | [1,H,W] | [H,W,1] | [1,1,H,W] | [1,H,W,1]
    Returns a dict with angular error mean/median and accuracy at thresholds.
    """
    # Convert pred/target to [H,W,3]
    def to_hw3(x):
        x = x.to(torch.float32)
        if x.dim() == 4:
            if x.size(0) == 1 and x.size(1) == 3:
                x = x.squeeze(0).permute(1, 2, 0)
            elif x.size(0) == 1 and x.size(-1) == 3:
                x = x.squeeze(0)
            else:
                raise ValueError(f"Unsupported 4D normal shape {tuple(x.shape)}")
        elif x.dim() == 3:
            if x.size(0) == 3:
                x = x.permute(1, 2, 0)
            elif x.size(-1) == 3:
                pass
            else:
                raise ValueError(f"Unsupported 3D normal shape {tuple(x.shape)}")
        else:
            raise ValueError(f"Normal must be 3D or 4D, got {x.dim()}D")
        return x

    # Convert mask to [H,W] boolean
    def to_hw_mask(m, hw):
        m = m.to(torch.float32)
        while m.dim() > 2 and m.size(0) == 1:
            m = m.squeeze(0)
        if m.dim() == 3:
            if m.size(-1) == 1:
                m = m.squeeze(-1)
            elif m.size(0) == 1:
                m = m.squeeze(0)
            else:
                raise ValueError(f"Unsupported mask shape {tuple(m.shape)}")
        if m.dim() != 2:
            raise ValueError(f"Mask must be [H,W], got {tuple(m.shape)}")
        H, W = hw
        if m.shape != (H, W):
            raise ValueError(f"Mask shape {tuple(m.shape)} != normal shape {(H, W)}")
        return (m > 0)

    # Standardize shapes
    pred_hw3 = to_hw3(pred)
    target_hw3 = to_hw3(target)
    if pred_hw3.shape != target_hw3.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred_hw3.shape)} vs {tuple(target_hw3.shape)}")
    H, W, _ = pred_hw3.shape
    valid = to_hw_mask(mask, (H, W))

    # No valid pixels
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {k: float('nan') for k in [
            'angular_error_mean',
            'angular_error_median',
            'within_11_point_5_deg',
            'within_22_point_5_deg',
            'within_30_deg'
        ]}

    # Extract valid pixels
    pred_v = pred_hw3[valid].reshape(n_valid, 3)
    target_v = target_hw3[valid].reshape(n_valid, 3)

    # Normalize
    eps = 1e-6
    pred_unit = pred_v / pred_v.norm(dim=1, keepdim=True).clamp_min(eps)
    target_unit = target_v / target_v.norm(dim=1, keepdim=True).clamp_min(eps)

    # Angular error
    dot = (pred_unit * target_unit).sum(dim=1).clamp(-1.0, 1.0)
    ang = torch.acos(dot) * (180.0 / torch.pi)

    # Metrics
    mean_err = ang.mean()
    median_err = ang.median()
    p11_5 = (ang < 11.5).float().mean() * 100.0
    p22_5 = (ang < 22.5).float().mean() * 100.0
    p30 = (ang < 30.0).float().mean() * 100.0

    return {
        'angular_error_mean': float(mean_err.item()),
        'angular_error_median': float(median_err.item()),
        'within_11_point_5_deg': float(p11_5.item()),
        'within_22_point_5_deg': float(p22_5.item()),
        'within_30_deg': float(p30.item())
    }