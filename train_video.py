import argparse
import logging
import os
import time
import datetime
import warnings
from pathlib import Path
import torch

from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter

from datasets.HumanVideo import HumanVideoDataset
from losses.loss_video import VideoLoss, mask_bce_loss

from utils.helper import set_seed, build_optimizer, save_checkpoint
from utils.logging import init_log, DummyLogger
from utils.dinov3_helper import get_model_config, load_model_checkpoint
from model.human_temporal import Human

warnings.filterwarnings("ignore")

def load_static_init(dynamic_model: torch.nn.Module, static_ckpt: str, logger):
    if static_ckpt and os.path.exists(static_ckpt):
        load_model_checkpoint(dynamic_model, static_ckpt, logger)
    else:
        logger.info("[Init] No static checkpoint provided, skip.")

def freeze_encoder(model: torch.nn.Module, logger):
    name = []
    for n, p in model.named_parameters():
        if n.startswith("pretrained."):
            p.requires_grad = False
            name.append(n)
    logger.info(f"[Freeze] Encoder params frozen: {name}")

def main():
    release_root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default="vitl", choices=['vitb', 'vitl'])
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--bs", type=int, default=1, help="clips per GPU")
    ap.add_argument("--frames", type=int, default=32, help="T frames per clip")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--save-path", type=str, default=str(release_root / "outputs" / "dynamic_vitl"))
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--mixed_precision", action="store_true")

    ap.add_argument("--video-list", type=str, required=True, help="filelist of dynamic samples")
    ap.add_argument("--data-root", type=str, required=True, help="root dir containing image/depth/normal/mask folders")
    ap.add_argument("--cse-config", type=str, default=None, help='DensePose CSE config yaml')
    ap.add_argument("--cse-weights", type=str, default=None, help='DensePose CSE model weights')
    ap.add_argument("--cse-embedder", type=str, default=None, help='DensePose CSE embedder weights')

    ap.add_argument("--max_iters", type=int, default=35000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)

    ap.add_argument("--static_ckpt", type=str, default="", help="static model checkpoint")
    ap.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True, help="freeze encoder parameters under pretrained.*")

    ap.add_argument("--normal_weight", type=float, default=0.1)
    ap.add_argument("--mask_weight", type=float, default=0.05)

    args = ap.parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision='fp16' if args.mixed_precision else 'no',
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    if accelerator.is_main_process:
        os.makedirs(args.save_path, exist_ok=True)
        writer = SummaryWriter(log_dir=args.save_path)
        logger = init_log('global', logging.INFO); logger.propagate = False
        writer.add_text('hparams', str(vars(args)))
    else:
        writer = None; logger = DummyLogger()

    size = (args.img_size, args.img_size)
    trainset = HumanVideoDataset(filelist_path=args.video_list, data_root=args.data_root, size=size, clip_length=args.frames, stride=args.stride, mode="train", sampling="random")
    train_loader_kwargs = {
        "batch_size": args.bs,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 4
    trainloader = DataLoader(trainset, **train_loader_kwargs)

    model = Human(
        **get_model_config(args.encoder, num_frames=args.frames),
        cse_config=args.cse_config,
        cse_weights=args.cse_weights,
        cse_embedder=args.cse_embedder,
    )
    load_static_init(model, args.static_ckpt, logger)

    if args.freeze_backbone:
        freeze_encoder(model, logger)

    optimizer_config = {
        "type": "AdamW",
        "params": [
            {
                "params": {"include": ["*tm_l3*", "*tm_l4*", "*tm_p4*", "*tm_p3*"], "exclude": ["*pretrained*"]},
                "lr": 1e-4, "weight_decay": 0.05, "betas": (0.9,0.95),
            },
            {
                "params": {"include": ["*"], "exclude": ["*tm_l3*", "*tm_l4*", "*tm_p4*", "*tm_p3*", "*pretrained*"]},
                "lr": 1e-6, "weight_decay": 0.05, "betas": (0.9,0.95),
            },
        ]
    }
    optimizer = build_optimizer(model, optimizer_config)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/100, total_iters=2000)
    poly = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_iters-2000, power=1.5)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, poly], milestones=[2000])

    videoloss = VideoLoss(device=accelerator.device).to(accelerator.device)
    model, optimizer, trainloader, lr_scheduler = accelerator.prepare(model, optimizer, trainloader, lr_scheduler)

    model.train()
    iter_count = 0
    t0 = time.time()
    logger.info("--- Starting Video Training ---")
    while iter_count < args.max_iters:
        for sample in trainloader:
            imgs = sample["image"]
            depth_gt = sample["depth_relative"]
            mask_gt = sample["valid_mask"]
            normal_gt = sample['normal']

            with accelerator.accumulate(model):
                B,T,H,W = depth_gt.shape
                outputs = model(imgs)
                ssi, n_loss, flw_d, flw_n = videoloss(
                    outputs["depth"].reshape(B, T, H, W), 
                    depth_gt, 
                    outputs["normal"].reshape(B, T, 3, H, W), 
                    normal_gt, 
                    mask_gt, 
                    imgs)
                m_loss = mask_bce_loss(outputs["mask"], mask_gt.reshape(-1, H, W), (~mask_gt.reshape(-1, H, W))) * args.mask_weight
                n_loss = n_loss * args.normal_weight

                total = (ssi + n_loss + flw_d + flw_n + m_loss).mean()
                accelerator.backward(total)
                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            iter_count += 1

            if iter_count % args.log_every == 0 and accelerator.is_main_process:
                elapsed = time.time()-t0
                eta = elapsed/iter_count*(args.max_iters-iter_count)
                logger.info(
                    f"Iter {iter_count}/{args.max_iters} | "
                    f"loss: {total.item():.4f} | "
                    f"ssi: {ssi.item():.4f} | "
                    f"n_loss: {n_loss.item():.4f} | "
                    f"flw_d: {flw_d.item():.4f} | "
                    f"flw_n: {flw_n.item():.4f} | "
                    f"m_loss: {m_loss.item():.4f} | "
                    f"ETA: {str(datetime.timedelta(seconds=int(eta)))}")
                
                current_lrs = [g['lr'] for g in optimizer.param_groups]
                if writer:
                    writer.add_scalar("train/total_loss", total.item(), iter_count)
                    writer.add_scalar("train/depth_loss", ssi.item(), iter_count)
                    writer.add_scalar("train/normal_loss", float(n_loss), iter_count)
                    writer.add_scalar("train/flw_d_loss", flw_d.item(), iter_count)
                    writer.add_scalar("train/flw_n_loss", flw_n.item(), iter_count)
                    writer.add_scalar("train/mask_loss", float(m_loss), iter_count)
                    for i, lr in enumerate(current_lrs):
                        writer.add_scalar(f'lr/group_{i}', lr, iter_count)

            if iter_count % args.eval_every == 0 or iter_count == args.max_iters:
                if accelerator.is_main_process:
                    save_checkpoint(accelerator, model, iter_count, args, logger)
                accelerator.wait_for_everyone()

            if iter_count >= args.max_iters:
                break

    logger.info("--- Training Finished ---")
    if accelerator.is_main_process and writer: 
        writer.close()

if __name__ == "__main__":
    main()
