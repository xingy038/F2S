import torch
import torch.nn as nn
import torch.nn.functional as F

class RAFTFlow:
    def __init__(self, device=None):
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=True).to(self.device).eval()
        self.tf = self.weights.transforms()

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def pair_flow(self, img0, img1, size=None):
        if size is not None:
            img0 = F.interpolate(img0, size=size, mode='bilinear', align_corners=True)
            img1 = F.interpolate(img1, size=size, mode='bilinear', align_corners=True)

        i0, i1 = self.tf(img0, img1)
        list_of_flows = self.model(i0.to(self.device), i1.to(self.device))  # list of (B,2,H,W)
        flow = list_of_flows[-1]
        return flow

    @torch.no_grad()
    def sequence_flows(self, images, size=None):
        B, T, _, _, _ = images.shape
        F_fwd, F_bwd = [], []
        for t in range(T-1):
            f = self.pair_flow(images[:, t],   images[:, t+1], size=size)
            b = self.pair_flow(images[:, t+1], images[:, t],   size=size)
            F_fwd.append(f)
            F_bwd.append(b)
        return F_fwd, F_bwd

def _make_coords_grid(B, H, W, device):
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((x, y), dim=0).float()
    return grid.unsqueeze(0).repeat(B,1,1,1)  # [B,2,H,W]

def warp_by_flow(x, flow):
    B, C, H, W = x.shape
    base = _make_coords_grid(B, H, W, x.device)
    tgt  = base + flow
    gx = 2.0 * (tgt[:,0] / max(W-1,1)) - 1.0
    gy = 2.0 * (tgt[:,1] / max(H-1,1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1)  # [B,H,W,2]
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

def cycle_mask(F_fwd, F_bwd, tau_pix=1.5):
    Fbwd_w = warp_by_flow(F_bwd, F_fwd)
    err = (F_fwd + Fbwd_w).pow(2).sum(1, keepdim=True).sqrt()
    return (err < tau_pix).float()

def resize_flow(flow, Ht, Wt, Hs, Ws):
    sx, sy = Wt/float(Ws), Ht/float(Hs)
    f = F.interpolate(flow, size=(Ht, Wt), mode='bilinear', align_corners=True)
    f[:,0] *= sx; f[:,1] *= sy
    return f

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based, frame_id_mask=None):
    # mask for distinguish different frames
    valid_id_mask_x = torch.ones_like(mask[:, :, 1:])
    valid_id_mask_y = torch.ones_like(mask[:, 1:, :])
    if frame_id_mask is not None:
        valid_id_mask_x = ((frame_id_mask[:, :, 1:] - frame_id_mask[:, :, :-1]) == 0).to(mask.dtype)
        valid_id_mask_y = ((frame_id_mask[:, 1:, :] - frame_id_mask[:, :-1, :]) == 0).to(mask.dtype)
    
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(torch.mul(mask[:, :, 1:], mask[:, :, :-1]), valid_id_mask_x)
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(torch.mul(mask[:, 1:, :], mask[:, :-1, :]), valid_id_mask_y)
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


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

def normalize_prediction_robust(target, mask, ms=None):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    if ms is None:
        m = torch.zeros_like(ssum)
        s = torch.ones_like(ssum)

        m[valid] = torch.median((mask[valid] * target[valid]).view(valid.sum(), -1), dim=1).values
    else:
        m, s = ms

    target = target - m.view(-1, 1, 1)

    if ms is None:
        sq = torch.sum(mask * target.abs(), (1, 2))
        s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1)), (m.detach(), s.detach())

class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=2.0, scales=4, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask, num_frame_h=1, no_norm=True):
        if no_norm:
            scale, shift = compute_scale_and_shift(prediction, target, mask)
            self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        else:
            self.__prediction_ssi, self.__prediction_median_scale = normalize_prediction_robust(prediction, mask)
            target, self.__target_median_scale = normalize_prediction_robust(target, mask)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target, mask, num_frame_h=num_frame_h
            )

        return total

    def get_median_scale(self):
        return self.__prediction_median_scale, self.__target_median_scale

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class TrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.trim = trim

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask, weight_mask=None):
        if torch.sum(mask) == 0:
            return torch.sum(prediction) * 0.0
        M = torch.sum(mask, (1, 2))
        res = prediction - target
        if weight_mask is not None:
            res = res * weight_mask
        res = res[mask.bool()].abs()
        trimmed, _ = torch.sort(res.view(-1), descending=False)
        keep_num = int(len(res) * (1.0 - self.trim))
        if keep_num <= 0:
            return torch.sum(prediction) * 0.0
        trimmed = trimmed[: keep_num]

        return self.__reduction(trimmed, M)

    
class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask, num_frame_h=1):
        total = 0

        frame_id_mask = None
        if num_frame_h > 1:
            frame_h = mask.shape[1] // num_frame_h
            frame_id_mask = torch.zeros_like(mask)
            for i in range(num_frame_h):
                frame_id_mask[:, i*frame_h:(i+1)*frame_h, :] = i+1

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
                frame_id_mask=frame_id_mask[:, ::step, ::step] if num_frame_h > 1 else None,
            )

        return total

    
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

class VideoLoss(nn.Module):
    def __init__(self,
                 depth_alpha=1.0, depth_scales=4, depth_trim=0.0, reduction="batch-based", eps=1e-6,
                 lambda_flow_depth=1.0, lambda_flow_normal=0.1, tau_pix=1.5, use_edge_mask=True,
                 raft_resize_to_8x: bool = True,
                 device: str = None):
        super().__init__()
        self.depth_spatial  = TrimmedProcrustesLoss(alpha=depth_alpha, scales=depth_scales, trim=depth_trim, reduction=reduction)
        self.normal_spatial = NormalHFMultiScaleLoss()
        self.lambda_flow_depth  = lambda_flow_depth
        self.lambda_flow_normal = lambda_flow_normal
        self.tau_pix = tau_pix
        self.use_edge_mask = use_edge_mask
        self._reduce = reduction_batch_based if reduction == "batch-based" else reduction_image_based
        self.eps = eps

        self._raft = RAFTFlow(device=device)
        self.raft_resize_to_8x = raft_resize_to_8x

    @torch.no_grad()
    def _seq_flows(self, images):
        B, T, C, H, W = images.shape
        if self.raft_resize_to_8x:
            H8 = int((H + 7)//8)*8; W8 = int((W + 7)//8)*8
            F_fwd, F_bwd = self._raft.sequence_flows(images, size=(H8, W8))
            if (H8, W8) != (H, W):
                F_fwd = [resize_flow(f, H, W, H8, W8) for f in F_fwd]
                F_bwd = [resize_flow(b, H, W, H8, W8) for b in F_bwd]
        else:
            F_fwd, F_bwd = self._raft.sequence_flows(images, size=None)
        return F_fwd, F_bwd

    def _non_edge_mask_from_depth(self, D, k=3):
        gx = D[..., :, 1:] - D[..., :, :-1]
        gy = D[..., 1:, :] - D[..., :-1, :]
        G = torch.zeros_like(D)
        G[..., :, 1:] += gx.abs(); G[..., 1:, :] += gy.abs()
        th = G.mean(dim=(-2,-1), keepdim=True)
        m = (G <= th).float()
        if k > 0:
            m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=2*k+1, stride=1, padding=k)
        return m

    def _flow_stab_depth(self, D_pred, F_fwd_list, F_bwd_list, mask=None):
        B, T, H, W = D_pred.shape
        if T < 2: return D_pred.sum()*0.0
        losses = []
        for t in range(T-1):
            Dt   = D_pred[:, t].unsqueeze(1)
            Dtp1 = D_pred[:, t+1].unsqueeze(1)
            Ff, Fb = F_fwd_list[t], F_bwd_list[t]
            Mf = cycle_mask(Ff, Fb, self.tau_pix)
            Mb = cycle_mask(Fb, Ff, self.tau_pix)
            Dt_w   = warp_by_flow(Dt,   Ff)
            Dtp1_w = warp_by_flow(Dtp1, Fb)
            diff = (Dt_w - Dtp1).abs() + (Dtp1_w - Dt).abs()
            M = Mf * Mb
            if mask is not None:
                M = M * (mask[:, t:t+1].float() * mask[:, t+1:t+2].float())
            if self.use_edge_mask:
                M = M * self._non_edge_mask_from_depth(Dt) * self._non_edge_mask_from_depth(Dtp1)
            num = M.sum(dim=(1,2,3)).clamp_min(1.0)
            loss_pair = (diff*M).sum(dim=(1,2,3)) / num
            losses.append(loss_pair)
        losses = torch.stack(losses,0).mean(0)
        return self._reduce(losses, torch.ones_like(losses))

    def _flow_stab_normal(self, N_pred, F_fwd_list, F_bwd_list, mask=None):
        B, T, C, H, W = N_pred.shape
        if T < 2: return N_pred.sum()*0.0
        N_pred = F.normalize(N_pred, dim=2, eps=self.eps)
        losses = []
        for t in range(T-1):
            Nt, Ntp1 = N_pred[:, t], N_pred[:, t+1]
            Ff, Fb = F_fwd_list[t], F_bwd_list[t]
            Mf = cycle_mask(Ff, Fb, self.tau_pix)
            Mb = cycle_mask(Fb, Ff, self.tau_pix)
            M  = Mf * Mb
            if mask is not None:
                M = M * (mask[:, t:t+1].float() * mask[:, t+1:t+2].float())
            Nt_w   = warp_by_flow(Nt,   Ff)
            Ntp1_w = warp_by_flow(Ntp1, Fb)
            cos_f = (F.normalize(Nt_w,   dim=1, eps=self.eps) * Ntp1).sum(1, keepdim=True).clamp(-1,1)
            cos_b = (F.normalize(Ntp1_w, dim=1, eps=self.eps) * Nt  ).sum(1, keepdim=True).clamp(-1,1)
            diff = (1.0 - cos_f) + (1.0 - cos_b)
            num = M.sum(dim=(1,2,3)).clamp_min(1.0)
            loss_pair = (diff*M).sum(dim=(1,2,3)) / num
            losses.append(loss_pair)
        losses = torch.stack(losses,0).mean(0)
        return self._reduce(losses, torch.ones_like(losses))

    def forward(self, depth_pred, depth_gt, normal_pred, normal_gt, mask, images):
        """
        depth_pred:  [B,T,H,W]
        normal_pred: [B,T,3,H,W]
        images:      [B,T,3,H,W]
        """
        B, T, H, W = depth_gt.shape
        ssi = self.depth_spatial(prediction=depth_pred.flatten(0, 1), target=depth_gt.flatten(0, 1), mask=mask.flatten(0, 1).float())
        scale, shift = compute_scale_and_shift(depth_pred.flatten(1, 2), depth_gt.flatten(1, 2), mask.flatten(1, 2))
        depth_pred = scale.view(-1, 1, 1, 1) * depth_pred + shift.view(-1, 1, 1, 1)
        normal = self.normal_spatial(normal_pred.reshape(-1, 3, H, W), normal_gt.reshape(-1, 3, H, W), mask.reshape(-1, H, W))

        with torch.no_grad():
            F_fwd_list, F_bwd_list = self._seq_flows(images)

        flow_depth = self._flow_stab_depth(depth_pred,  F_fwd_list, F_bwd_list, mask=mask)
        flow_normal = self._flow_stab_normal(normal_pred, F_fwd_list, F_bwd_list, mask=mask)

        return ssi, normal, self.lambda_flow_depth*flow_depth, self.lambda_flow_normal*flow_normal
