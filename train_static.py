import os
import time
import logging
import argparse
import datetime
import warnings
from pathlib import Path
import numpy as np
import torch

from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedDataParallelKwargs

from torch.utils.data import WeightedRandomSampler
from datasets.Human import HumanDataset
from datasets.SynthHuman import HumanDataset_SynthHuman
from model.human import Human
from losses.loss import ScaleAndShiftInvariantLoss, NormalHFMultiScaleLoss, mask_bce_loss
from utils.metric import eval_depth_with_scale, eval_normal
from utils.logging import init_log, DummyLogger
from utils.dinov3_helper import get_model_config, load_pretrained_backbone
from vis.utils import save_colormap_depth, save_normal_map, save_mask
from utils.helper import set_seed, infinite_loader, build_optimizer, save_checkpoint

warnings.filterwarnings("ignore")

def main():
    release_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Depth Estimation Training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--encoder', default='vitl', choices=['vitb', 'vitl'], help="DINOv3 backbone")
    parser.add_argument('--img-size', default=896, type=int, help="Input image size")
    parser.add_argument('--max_iters', default=50000, type=int, help="Total number of training iterations")
    parser.add_argument('--bs', default=8, type=int, help="Batch size per GPU")
    parser.add_argument('--save-path', type=str, default=str(release_root / 'outputs' / 'static_vitl'), help="Path for saving logs and checkpoints")
    parser.add_argument('--pretrained-backbone', type=str, default=str(release_root / 'checkpoints' / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'), help="Path to DINOv3 backbone weights")
    parser.add_argument('--train-list', type=str, required=True, help='Training split file for the real human dataset')
    parser.add_argument('--train-root', type=str, required=True, help='Root directory for the real human dataset')
    parser.add_argument('--synth-list', type=str, default='', help='Optional training split file for SynthHuman')
    parser.add_argument('--synth-root', type=str, default='', help='Root directory for SynthHuman')
    parser.add_argument('--val-list', type=str, required=True, help='Validation split file')
    parser.add_argument('--val-root', type=str, required=True, help='Root directory for the validation dataset')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training (fp16)')
    parser.add_argument('--num_vis_images', type=int, default=20, help='Number of validation images to visualize')
    parser.add_argument('--log_every_iters', type=int, default=50, help='Log metrics every N iterations')
    parser.add_argument('--eval_every_iters', type=int, default=1000, help='Evaluate and save checkpoint every N iterations')
    parser.add_argument('--cse-config', type=str, default=None, help='DensePose CSE config yaml')
    parser.add_argument('--cse-weights', type=str, default=None, help='DensePose CSE model weights')
    parser.add_argument('--cse-embedder', type=str, default=None, help='DensePose CSE embedder weights')
    parser.add_argument('--freeze-backbone', action='store_true', help="Freeze ViT backbone (parameters under 'pretrained.' prefix)")
    parser.add_argument('--mask-weight', type=float, default=0.05, help="mask BCE weight")
    parser.add_argument('--normal-weight', type=float, default=0.08, help="normal weight")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    args = parser.parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision='fp16' if args.mixed_precision else 'no',
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.save_path, exist_ok=True)
        tb_dir = args.save_path
        writer = SummaryWriter(log_dir=tb_dir)
        logger = init_log('global', logging.INFO)
        logger.propagate = False
        writer.add_text('hparams', str(vars(args)))
    else:
        writer = None
        logger = DummyLogger()

    if bool(args.synth_list) != bool(args.synth_root):
        raise ValueError("Please provide both --synth-list and --synth-root, or leave both empty.")

    size = (args.img_size, args.img_size)
    trainset_1 = HumanDataset(args.train_list, args.train_root, 'train', size=size)
    trainset = trainset_1
    sampler = None
    if args.synth_list:
        trainset_2 = HumanDataset_SynthHuman(args.synth_list, args.synth_root, 'train', size=size)
        len1, len2 = len(trainset_1), len(trainset_2)
        weights = [1.0 / len1] * len1 + [1.0 / len2] * len2
        sampler = WeightedRandomSampler(weights, num_samples=len1 + len2, replacement=True)
        trainset = ConcatDataset([trainset_1, trainset_2])
    valset = HumanDataset(args.val_list, args.val_root, 'val', size=size)

    train_loader_kwargs = {
        'batch_size': args.bs,
        'pin_memory': True,
        'num_workers': args.num_workers,
        'persistent_workers': args.num_workers > 0,
        'drop_last': True,
    }
    val_loader_kwargs = {
        'batch_size': 1,
        'pin_memory': True,
        'num_workers': args.num_workers,
        'persistent_workers': args.num_workers > 0,
        'drop_last': False,
        'shuffle': False,
    }
    if args.num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = 4
        val_loader_kwargs['prefetch_factor'] = 4
    if sampler is None:
        train_loader_kwargs['shuffle'] = True
    else:
        train_loader_kwargs['sampler'] = sampler
    trainloader = DataLoader(trainset, **train_loader_kwargs)
    valloader = DataLoader(valset, **val_loader_kwargs)

    depth_loss = ScaleAndShiftInvariantLoss(alpha=1.0, scales=4, reduction='batch-based').to(accelerator.device)
    normal_loss = NormalHFMultiScaleLoss().to(accelerator.device)

    logger.info(f'Train samples: {len(trainset)}, Val samples: {len(valset)}')

    model = Human(
        **get_model_config(args.encoder),
        cse_config=args.cse_config,
        cse_weights=args.cse_weights,
        cse_embedder=args.cse_embedder,
    )
    if args.pretrained_backbone:
        load_pretrained_backbone(model, args.pretrained_backbone, logger)
    else:
        logger.info("No pretrained backbone provided. Training will start from random initialization.")

    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if n.startswith('pretrained.'):
                p.requires_grad = False

    optimizer_config = {
        "type": "AdamW",
        "params": [
            {"params": {"include": ["*"], "exclude": ["*pretrained*"]}, "lr": args.lr, "weight_decay": 0.05, "betas": (0.9, 0.95)},
            {"params": {"include": ["*pretrained*"], "exclude": []}, "lr": args.lr * 0.1, "weight_decay": 0.05, "betas": (0.9, 0.95)},
        ]
    }
    optimizer = build_optimizer(model, optimizer_config)
    
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/100, total_iters=2000)
    poly = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_iters-2000, power=1.5)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, poly], milestones=[2000])

    model, optimizer, trainloader, valloader, lr_scheduler = accelerator.prepare(
        model, optimizer, trainloader, valloader, lr_scheduler
    )

    train_iterator = infinite_loader(trainloader)
    
    iter_count = 0
    global_start_time = time.time()

    logger.info("--- Starting Training ---")
    while iter_count < args.max_iters:
        model.train()
        sample = next(train_iterator)
        
        with accelerator.accumulate(model):
            outputs = model(sample['image'])
            
            depth_gt, mask_gt = sample['depth_relative'], sample['valid_mask']

            d_loss = depth_loss(outputs['depth'], depth_gt, mask_gt)
            n_loss = normal_loss(outputs['normal'], sample['normal'], mask_gt) * args.normal_weight
            m_loss = mask_bce_loss(outputs['mask'], mask_gt, ~mask_gt) * args.mask_weight
            total_loss = (d_loss + m_loss + n_loss).mean()
            
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
        
        iter_count += 1

        if iter_count % args.log_every_iters == 0 and accelerator.is_main_process:
            elapsed = time.time() - global_start_time
            eta_seconds = (elapsed / iter_count) * (args.max_iters - iter_count)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            current_lrs = [g['lr'] for g in optimizer.param_groups]
            logger.info(f"Iter: {iter_count}/{args.max_iters} | Loss: {total_loss.item():.5f} | ETA: {eta_str}")
            
            if writer is not None:
                writer.add_scalar('train/total_loss', total_loss.item(), iter_count)
                writer.add_scalar('train/depth_loss', d_loss.mean().item(), iter_count)
                writer.add_scalar('train/normal_loss', n_loss.mean().item(), iter_count)
                writer.add_scalar('train/mask_loss', m_loss.mean().item(), iter_count)
                for i, lr in enumerate(current_lrs):
                    writer.add_scalar(f'lr/group_{i}', lr, iter_count)

        if iter_count % args.eval_every_iters == 0 or iter_count == args.max_iters:
            model.eval()
            all_depth_metrics = []
            all_normal_metrics = []
            vis_path = os.path.join(args.save_path, f'vis/iter_{iter_count}')
            if accelerator.is_main_process:
                os.makedirs(vis_path, exist_ok=True)

            with torch.no_grad():
                for i, val_sample in enumerate(valloader):
                    meta = val_sample['meta']
                    pad_x, pad_y = meta['pad']
                    orig_h, orig_w = meta['orig_size']

                    pred_val = accelerator.unwrap_model(model)(val_sample['image'])
                    
                    depth_full = pred_val['depth']
                    normal_full = pred_val['normal']
                    mask_full = pred_val['mask']
                    full_h, full_w = depth_full.shape[1:]

                    h_sc = full_h - 2 * pad_y
                    w_sc = full_w - 2 * pad_x

                    depth_crop = depth_full[..., pad_y: pad_y + h_sc, pad_x: pad_x + w_sc]
                    normal_crop = normal_full[..., pad_y: pad_y + h_sc, pad_x: pad_x + w_sc]
                    mask_crop = mask_full[..., pad_y: pad_y + h_sc, pad_x: pad_x + w_sc]

                    pred_depth = F.interpolate(depth_crop.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=True)[0]
                    pred_normal = F.interpolate(normal_crop, size=(orig_h, orig_w), mode='bilinear', align_corners=True)[0]
                    pred_normal = F.normalize(pred_normal, dim=0, eps=1e-6)
                    pred_mask = F.interpolate(mask_crop.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=True)[0]

                    gt_depth = val_sample['depth']
                    gt_mask = val_sample['valid_mask']
                    gt_normal = val_sample['normal'][0]

                    depth_metrics = eval_depth_with_scale(pred_depth, gt_depth, gt_mask)
                    normal_metrics = eval_normal(pred_normal, gt_normal, gt_mask)
                    all_depth_metrics.append(depth_metrics)
                    all_normal_metrics.append(normal_metrics)

                    if i < args.num_vis_images and accelerator.is_main_process:
                        save_colormap_depth(pred_depth, gt_mask, os.path.join(vis_path, f'depth_{i:04d}.png'))
                        save_normal_map(pred_normal, gt_mask, os.path.join(vis_path, f'normal_{i:04d}.png'))
                        save_mask(pred_mask, os.path.join(vis_path, f'mask_{i:04d}.png'))

            if accelerator.is_main_process:
                avg_depth  = {k: np.mean([m[k] for m in all_depth_metrics ]) for k in all_depth_metrics[0]}
                avg_normal = {k: np.mean([m[k] for m in all_normal_metrics]) for k in all_normal_metrics[0]}

                depth_items = sorted(avg_depth.items(), key=lambda x: x[0])
                max_name_len_d = max(len(name) for name, _ in depth_items) + 1
                logger.info("=" * 50)
                logger.info(f"Validation Iter={iter_count} Depth Metrics")
                for name, val in depth_items:
                    logger.info(f"{name:<{max_name_len_d}}{val:>8.4f}")
                logger.info("=" * 50)

                normal_items = sorted(avg_normal.items(), key=lambda x: x[0])
                max_name_len_n = max(len(name) for name, _ in normal_items) + 1
                logger.info(f"Validation Iter={iter_count} Normal Metrics")
                for name, val in normal_items:
                    logger.info(f"{name:<{max_name_len_n}}{val:>8.4f}")
                logger.info("=" * 50)

                if writer is not None:
                    for k, v in avg_depth.items():
                        writer.add_scalar(f"eval_depth/{k}", v, iter_count)
                    for k, v in avg_normal.items():
                        writer.add_scalar(f"eval_normal/{k}", v, iter_count)

                save_checkpoint(accelerator, model, iter_count, args, logger)

            accelerator.wait_for_everyone()

    logger.info("--- Training Finished ---")
    if accelerator.is_main_process and writer is not None:
        writer.close()

if __name__ == '__main__':
    main()
