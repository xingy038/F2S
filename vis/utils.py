from typing import *

import numpy as np
import matplotlib
import imageio
import matplotlib.cm as cm

def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal

def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored

def colorize_depth_affine(depth: np.ndarray, mask: np.ndarray = None, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        depth = np.where(mask, depth, np.nan)

    min_depth, max_depth = np.nanquantile(depth, 0.001), np.nanquantile(depth, 0.999)
    depth = (depth - min_depth) / (max_depth - min_depth)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](depth)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_disparity(disparity: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        disparity = np.where(mask, disparity, np.nan)
    
    if normalize:
        min_disp, max_disp = np.nanquantile(disparity, 0.001), np.nanquantile(disparity, 0.999)
        disparity = (disparity - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disparity)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored

def colorize_error_map(error_map: np.ndarray, mask: np.ndarray = None, cmap: str = 'plasma', value_range: Tuple[float, float] = None):
    vmin, vmax = value_range if value_range is not None else (np.nanmin(error_map), np.nanmax(error_map))
    cmap = matplotlib.colormaps[cmap]
    colorized_error_map = cmap(((error_map - vmin) / (vmax - vmin)).clip(0, 1))[..., :3]
    if mask is not None:
        colorized_error_map = np.where(mask[..., None], colorized_error_map, 0)
    colorized_error_map = np.ascontiguousarray((colorized_error_map.clip(0, 1) * 255).astype(np.uint8))
    return colorized_error_map

def save_colormap_depth(pred_depth, mask, output_path):
    """Saves a colormapped depth image."""
    depth_norm = pred_depth.cpu().numpy()
    mask = mask.cpu().numpy()
    valid_pixels = depth_norm[mask]
    if valid_pixels.size == 0:
        print(f"Warning: No valid pixels in mask for {output_path}, skipping.")
        return
    
    dmin, dmax = np.nanmin(valid_pixels), np.nanmax(valid_pixels)
    if dmax - dmin < 1e-6:
        print(f"Warning: Depth range is too small for {output_path}, skipping.")
        return
        
    depth_norm = (depth_norm - dmin) / (dmax - dmin)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    cmap = cm.get_cmap('Spectral_r')
    colored = cmap(depth_norm)
    rgb = (colored[..., :3] * 255).astype(np.uint8)
    mask_3c = np.repeat(mask[:, :, :, None], 3, axis=3)
    rgb[~mask_3c] = 0
    imageio.imwrite(output_path, rgb[0])

def save_normal_map(pred_normal, mask, output_path):
    mask = mask.cpu().numpy()
    if pred_normal.dim() == 3:
        pred_normal = pred_normal.unsqueeze(0)
    
    if pred_normal.dim() == 4:
        pred_normal = pred_normal[0]
    else:
        raise ValueError(f"Expected pred_normal to have 3 or 4 dims, got {pred_normal.dim()}")
    
    normal_np = pred_normal.cpu().numpy().transpose(1, 2, 0)
    normal_vis = np.clip((normal_np + 1.0) / 2.0, 0.0, 1.0)
    normal_uint8 = (normal_vis * 255).astype(np.uint8)
    mask_3c = np.repeat(mask[0][:, :, None], 3, axis=2)
    normal_uint8[~mask_3c] = 0
    
    imageio.imwrite(output_path, normal_uint8)


def save_mask(pred_mask, output_path, threshold=0.5):
    mask = pred_mask.detach().cpu().float()

    if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask = mask[0, 0]
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 2:
        pass
    else:
        if mask.ndim == 3:
            mask = mask[0]
        else:
            raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

    bin_mask = (mask > float(threshold)).numpy().astype(np.uint8) * 255
    imageio.imwrite(output_path, bin_mask)