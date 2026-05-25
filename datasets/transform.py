import cv2
import numpy as np
import math
import random
import torch
import torchvision.transforms.functional as TF
from typing import Dict, List, Tuple, Optional
    
class Resize_Pad(object):
    """Resize sample to given size, then optionally pad to a square."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        pad_to_square=True,
        pad_border_type=cv2.BORDER_REPLICATE,
    ):
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

        self.__pad_to_square = pad_to_square
        self.__pad_border_type = pad_border_type

        if pad_to_square and width != height:
            raise ValueError("When pad_to_square=True, width and height must be the same.")

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = int(np.round(x / self.__multiple_of) * self.__multiple_of)

        if max_val is not None and y > max_val:
            y = int(np.floor(x / self.__multiple_of) * self.__multiple_of)

        if y < min_val:
            y = int(np.ceil(x / self.__multiple_of) * self.__multiple_of)

        return y

    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def _pad_square(self, img, pad_val=0):
        h, w = img.shape[:2]
        if h == self.__height and w == self.__width:
            return img, (0, 0)
        pad_left = (self.__width - w) // 2
        pad_right = self.__width - w - pad_left
        pad_top = (self.__height - h) // 2
        pad_bottom = self.__height - h - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, borderType=self.__pad_border_type, value=pad_val,)
        return img, (pad_left, pad_top)

    def __call__(self, sample):
        h0, w0 = sample["image"].shape[:2]

        width, height = self.get_size(w0, h0)

        sx, sy = width / w0, height / h0

        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)
            if "normal" in sample:
                sample["normal"] = cv2.resize(sample["normal"], (width, height), interpolation=cv2.INTER_NEAREST)
            if "mask" in sample:
                sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            if "intrinsic" in sample:
                K = sample["intrinsic"].copy()
                K[0, 0] *= sx
                K[0, 2] *= sx
                K[1, 1] *= sy
                K[1, 2] *= sy
                sample["intrinsic"] = K

        if self.__pad_to_square:
            sample["image"], (pad_x, pad_y) = self._pad_square(sample["image"], 0)

            if self.__resize_target:
                if "depth" in sample:
                    sample["depth"], _ = self._pad_square(sample["depth"], 0)
                if "normal" in sample:
                    sample["normal"], _ = self._pad_square(sample["normal"], 0)
                if "mask" in sample:
                    sample["mask"], _ = self._pad_square(sample["mask"], 0)
                if "intrinsic" in sample:
                    K = sample["intrinsic"]
                    K[0, 2] += pad_x
                    K[1, 2] += pad_y
                    sample["intrinsic"] = K
        sample.setdefault("meta", {})
        sample["meta"]["pad"]  = (pad_x, pad_y)
        sample["meta"]["orig_size"] = (h0, w0)
        return sample

class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "normal" in sample:
            normal = np.transpose(sample["normal"], (2, 0, 1))
            sample["normal"] = np.ascontiguousarray(normal).astype(np.float32)

        return sample

class NormalizeDepth:
    def __init__(self, max_depth: float = 100.0):
        self.max_depth = max_depth

    def __call__(self, sample: dict) -> dict:
        depth = sample['depth'].astype(np.float32)
        mask  = sample['mask'].astype(bool)

        valid_mask = mask & (depth > 0.001) & (depth < self.max_depth)
        valid_depth = depth[valid_mask]

        low, high = np.percentile(valid_depth, (1, 99))
        valid_depth = np.clip(valid_depth, low, high)

        min, max = valid_depth.min(), valid_depth.max()
        depth_norm = np.zeros_like(depth, dtype=np.float32)
        depth_norm[valid_mask] = (valid_depth - min) / (max - min)
        sample['depth_relative'] = depth_norm
        return sample
    
class NormalizeDepthSequence:
    def __init__(self, max_depth: float = 100.0, percentiles=(1, 99)):
        self.max_depth = max_depth
        self.percentiles = percentiles

    def __call__(self, clip_list: List[Dict]) -> List[Dict]:
        all_vals = []

        for smp in clip_list:
            depth = smp['depth'].astype(np.float32)
            mask  = smp['mask'].astype(bool)
            valid = mask & (depth > 0.001) & (depth < self.max_depth)
            if np.any(valid):
                all_vals.append(depth[valid])

        if len(all_vals) == 0:
            for smp in clip_list:
                smp['depth_relative'] = np.zeros_like(smp['depth'], dtype=np.float32)
            return clip_list

        all_vals = np.concatenate(all_vals, axis=0)

        low, high = np.percentile(all_vals, self.percentiles)
        low = float(low); high = float(high)
        all_vals = np.clip(all_vals, low, high)
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
        scale = max(vmax - vmin, 1e-6)

        for smp in clip_list:
            depth = smp['depth'].astype(np.float32)
            mask  = smp['mask'].astype(bool)
            valid = mask & (depth > 0.001) & (depth < self.max_depth)

            depth_clip = np.clip(depth, low, high)
            depth_rel = np.zeros_like(depth, dtype=np.float32)
            depth_rel[valid] = (depth_clip[valid] - vmin) / scale
            smp['depth_relative'] = depth_rel

        return clip_list

class NormalizeNormal(object):
    """
    Normalize depth using the provided mask.
    """
    def __call__(self, sample: dict) -> dict:
        background_val = 0
        normal = sample['normal']
        mask = sample['mask']
        normal = normal / 255.0
        normal = 2.0 * normal - 1.0
        valid = mask > 0

        normal[~valid] = background_val
        
        sample['normal'] = normal
        return sample


class RandomFlip(object):
    """Randomly flip image, depth, mask, normal horizontally.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            for key in ["image", "depth", "mask", "normal"]:
                if key in sample:
                    sample[key] = cv2.flip(sample[key], 1)
                    if key == "normal":
                        sample["normal"][..., 0] *= -1.0
        return sample


class Jittering(object):
    def __init__(
        self,
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3),
        saturation=(0.7, 1.3),
        hue_delta=0.1,
        gamma=(0.7, 1.3),
    ):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue_delta  = hue_delta
        self.gamma      = gamma

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]
        t = torch.from_numpy(img).permute(2,0,1).float()
        t = TF.adjust_brightness(t, random.uniform(*self.brightness))
        t = TF.adjust_contrast(  t, random.uniform(*self.contrast))
        t = TF.adjust_saturation(t, random.uniform(*self.saturation))
        t = TF.adjust_hue(       t, random.uniform(-self.hue_delta, self.hue_delta))
        t = TF.adjust_gamma(     t, random.uniform(*self.gamma))
        t = t.clamp(0.0, 1.0)
        img_jit = t.permute(1, 2, 0).numpy().astype(np.float32)
        sample["image"] = img_jit
        return sample

# ----------------------------------------------------------------------
class Blurring(object):
    def __init__(self, blur_prob=0.5, kernel_size_range=(3,7)):
        self.prob = blur_prob
        self.kernel_size_range = kernel_size_range

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]
        if random.random() < self.prob:
            k = random.randint(*self.kernel_size_range)
            if k % 2 == 0:
                k += 1
            img = cv2.GaussianBlur(img, (k, k), 0)
        sample["image"] = img
        return sample

# ----------------------------------------------------------------------
class ShotNoise(object):
    def __init__(self, shot_noise_prob=0.5, k_range=(100,10000), sigma_range=(0.0, 0.02)):
        self.prob = shot_noise_prob
        self.k_range = k_range
        self.sigma_range = sigma_range

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]
        if random.random() < self.prob:
            logk = random.uniform(math.log(self.k_range[0]),
                                  math.log(self.k_range[1]))
            k = math.exp(logk)
            poisson = np.random.poisson(img * k) / k - img

            sigma = random.uniform(*self.sigma_range)
            gauss = np.random.normal(0.0, sigma, img.shape)

            img = np.clip(img + poisson + gauss, 0.0, 1.0)
        sample["image"] = img
        return sample

class JPEGCompression(object):
    def __init__(self, prob=0.5, quality_range=(20, 100)):
        self.prob = prob
        self.quality_range = quality_range

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]
        if random.random() < self.prob:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            q = random.randint(self.quality_range[0], self.quality_range[1])
            ok, enc = cv2.imencode('.jpg', img_u8, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = rgb.astype(np.float32) / 255.0
        sample["image"] = img
        return sample
    

class VideoConsistentJitter:

    def __init__(self,
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3),
        saturation=(0.7, 1.3),
        hue_delta=0.1,
        gamma=(0.7, 1.3),
        prob=1.0
    ):
        import torchvision.transforms.functional as TF
        self.TF = TF
        self.brightness = brightness; self.contrast = contrast
        self.saturation = saturation; self.hue_delta = hue_delta
        self.gamma = gamma; self.prob = prob

    def __call__(self, sample_list: List[Dict]) -> List[Dict]:
        if random.random() > self.prob:
            return sample_list
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        s = random.uniform(*self.saturation)
        h = random.uniform(-self.hue_delta, self.hue_delta)
        g = random.uniform(*self.gamma)
        out = []
        for smp in sample_list:
            img = smp["image"]
            t = torch.from_numpy(img).permute(2,0,1).float()
            t = self.TF.adjust_brightness(t, b)
            t = self.TF.adjust_contrast(  t, c)
            t = self.TF.adjust_saturation(t, s)
            t = self.TF.adjust_hue(       t, h)
            t = self.TF.adjust_gamma(     t, g)
            t = t.clamp(0.0, 1.0)
            smp = dict(smp); smp["image"] = t.permute(1,2,0).numpy().astype(np.float32)
            out.append(smp)
        return out
