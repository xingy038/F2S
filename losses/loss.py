import torch
import torch.nn.functional as F
from typing import *
import torch.nn as nn

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid]
                  * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)
        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
        
def mask_bce_loss(pred_mask_prob: torch.Tensor, gt_mask_pos: torch.Tensor, gt_mask_neg: torch.Tensor) -> torch.Tensor:
    loss = (gt_mask_pos | gt_mask_neg) * F.binary_cross_entropy(pred_mask_prob, gt_mask_pos.float(), reduction='none')
    loss = loss.mean()
    return loss

class NormalHFMultiScaleLoss(nn.Module):
    def __init__(self,
                 alpha: float = 1.0,
                 beta_grad: float = 0.4,
                 gamma_lap: float = 0.12,
                 edge_boost: float = 2.0,
                 strides=(1, 2, 4),
                 scale_weights=(1.0, 0.5, 0.25),
                 eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta_grad = beta_grad
        self.gamma_lap = gamma_lap
        self.edge_boost = edge_boost
        self.strides = strides
        self.scale_weights = scale_weights
        self.eps = eps

    @staticmethod
    def _sobel_kernels(device, dtype):
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=dtype, device=device).view(1,1,3,3)
        ky = torch.tensor([[1,2, 1],[0,0, 0],[-1,-2,-1]], dtype=dtype, device=device).view(1,1,3,3)
        return kx, ky

    @staticmethod
    def _laplacian_kernel(device, dtype):
        k = torch.tensor([[0, 1, 0],
                          [1,-4, 1],
                          [0, 1, 0]], dtype=dtype, device=device).view(1,1,3,3)
        return k

    def _downsample_normals_mask(self, n, m, stride):
        if stride == 1:
            return F.normalize(n, dim=1, eps=self.eps), m
        n_ds = F.avg_pool2d(n, kernel_size=stride, stride=stride)
        n_ds = F.normalize(n_ds, dim=1, eps=self.eps)
        m_ds = F.avg_pool2d(m.float().unsqueeze(1), kernel_size=stride, stride=stride).squeeze(1)
        m_ds = (m_ds > 0.5).to(m.dtype)
        return n_ds, m_ds

    def _edge_weight_from_gt(self, n_gt, valid, guide=None):
        device, dtype = n_gt.device, n_gt.dtype
        kx, ky = self._sobel_kernels(device, dtype)
        C = n_gt.shape[1]
        gx_g = F.conv2d(n_gt, kx.expand(C,1,3,3), padding=1, groups=C)
        gy_g = F.conv2d(n_gt, ky.expand(C,1,3,3), padding=1, groups=C)
        grad_mag = torch.sqrt(gx_g.pow(2) + gy_g.pow(2) + self.eps).mean(dim=1)  # [B,H,W]
        gm_min = grad_mag.amin(dim=(-2,-1), keepdim=True)
        gm_max = grad_mag.amax(dim=(-2,-1), keepdim=True)
        gm_n = (grad_mag - gm_min) / (gm_max - gm_min + 1e-8)
        edge_w = 1.0 + gm_n * (self.edge_boost - 1.0)
        edge_w = edge_w * valid.float() + (~valid).float()
        return edge_w

    def _reduce_map(self, x, valid, edge_w):
        w = edge_w * valid.float()
        denom = w.sum(dim=(-2, -1)).clamp(min=1.0)
        return ((x * w).sum(dim=(-2, -1)) / denom).mean()

    def forward(self, normals_pred: torch.Tensor, normals_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask[:, 0]
        mask = mask.to(dtype=torch.float32)

        n_pred_full = F.normalize(normals_pred, dim=1, eps=self.eps)
        n_gt_full   = F.normalize(normals_gt,   dim=1, eps=self.eps)

        device, dtype = n_gt_full.device, n_gt_full.dtype
        kx, ky = self._sobel_kernels(device, dtype)
        lap_k  = self._laplacian_kernel(device, dtype)

        total = 0.0
        for stride, w_s in zip(self.strides, self.scale_weights):
            n_pred_s, m_s = self._downsample_normals_mask(n_pred_full, mask, stride)
            n_gt_s,   _   = self._downsample_normals_mask(n_gt_full,   mask, stride)
            valid = (m_s > 0)

            cos = (n_pred_s * n_gt_s).sum(dim=1).clamp(-1.0, 1.0)
            ang_err = 1.0 - cos
            l1_map  = (n_pred_s - n_gt_s).abs().mean(dim=1)

            edge_w = self._edge_weight_from_gt(n_gt_s, valid)

            C = n_gt_s.shape[1]
            gx_p = F.conv2d(n_pred_s, kx.expand(C,1,3,3), padding=1, groups=C)
            gy_p = F.conv2d(n_pred_s, ky.expand(C,1,3,3), padding=1, groups=C)
            gx_g = F.conv2d(n_gt_s,   kx.expand(C,1,3,3), padding=1, groups=C)
            gy_g = F.conv2d(n_gt_s,   ky.expand(C,1,3,3), padding=1, groups=C)
            grad_err = (gx_p - gx_g).abs().mean(dim=1) + (gy_p - gy_g).abs().mean(dim=1)

            lap_p = F.conv2d(n_pred_s, lap_k.expand(C,1,3,3), padding=1, groups=C).mean(dim=1)
            lap_g = F.conv2d(n_gt_s,   lap_k.expand(C,1,3,3), padding=1, groups=C).mean(dim=1)
            lap_err = (lap_p - lap_g).abs()

            loss_cos = self._reduce_map(ang_err, valid, edge_w)
            loss_l1  = self._reduce_map(l1_map,  valid, edge_w)
            loss_grad= self._reduce_map(grad_err,valid, edge_w)
            loss_lap = self._reduce_map(lap_err, valid, edge_w)

            total = total + w_s * (loss_cos + self.alpha*loss_l1 + self.beta_grad*loss_grad + self.gamma_lap*loss_lap)

        return total
