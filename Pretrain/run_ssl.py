# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss
from models.ssl_head import SSLHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from dataset.dataloader import get_loader

from einops import rearrange


def main():
    def save_ckp(model, filename='ckpt.pt', optimizer=None, global_step=None, scheduler=None):
        state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
        save_dict = {"state_dict": state_dict, "global_step": global_step}

        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()

        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()

        filename = os.path.join(args.logdir, filename)
        torch.save(save_dict, filename)
        print("Saving checkpoint", filename)

    def train(args, global_step, train_loader, val_best, scaler):
        model.train()
        loss_train = []
        loss_train_recon = []
        loss_train_contrastive = []

        for step, batch in enumerate(train_loader):
            t1 = time()
            x = batch[0]["image"].cuda()
            y = batch[1]["image"].cuda()

            with autocast(enabled=args.amp):
                contrastive1_x, contrastive1_y, rec_x1 = model(x, y)
                if args.invis_patches:
                    x = model.swinViT.mask_layer.get_masked_gt(x)
                    rec_x1 = model.swinViT.mask_layer.get_masked_gt(rec_x1)
                loss, contrastive_loss, rec_loss = loss_function(contrastive1_x, contrastive1_y, rec_x1, x)

            loss_train.append(loss.item())
            loss_train_recon.append(rec_loss.item())
            loss_train_contrastive.append(contrastive_loss.item())

            writer.add_scalar("train/loss_total", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_recon", scalar_value=rec_loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_contrastive", scalar_value=contrastive_loss.item(), global_step=global_step)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()

            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss.item(), time() - t1))
            else:
                print("Step:{}/{}, Loss:{:.4f}, Rcloss:{:.4f}, Conloss:{:.4f},Time:{:.4f}".format(global_step,
                      args.num_steps, loss.item(), rec_loss.item(), loss.item() - rec_loss.item(), time() - t1))

            global_step += 1

            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                val_loss, val_loss_recon = validation(args, test_loader)
                writer.add_scalar("Validation/loss", scalar_value=val_loss.item(), global_step=global_step)
                writer.add_scalar("Validation/loss_recon", scalar_value=val_loss_recon.item(), global_step=global_step)
                writer.add_scalar("Validation/loss_recon", scalar_value=val_loss.item() - val_loss_recon.item(),
                                  global_step=global_step)

                if val_loss < val_best:
                    val_best = val_loss

                    save_ckp(model, "val_best.pt", optimizer=optimizer, global_step=global_step, scheduler=scheduler)
                    print(
                        "Best model was saved to {}! Best Val Loss: {:.4f}".format(
                            os.path.join(args.logdir, "valbest.pt"), val_best
                        )
                    )

            # We do not use iterable-style datasets class. So we need early stop the training if
            # the map dataloder still in loop
            if global_step > args.num_steps:
                break
            if global_step % 1000 == 0:
                save_ckp(model, f"{global_step}_ckpt.pt", optimizer=optimizer, global_step=global_step, scheduler=scheduler)
                print(
                    "Model was saved to {}!".format(
                        os.path.join(args.logdir, "final_ckpt.pt")
                    )
                )
        return global_step, loss.item(), val_best

    def validation(args, test_loader):
        model.eval()
        loss_val = []
        loss_val_recon = []

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                val_inputs = batch[0]["image"].cuda()
                val_inputs_y = batch[1]["image"].cuda()

                with autocast(enabled=args.amp):
                    contrastive1_x, contrastive1_y, rec_x1 = model(val_inputs, val_inputs_y)
                    if args.invis_patches:
                        val_inputs = model.swinViT.mask_layer.get_masked_gt(val_inputs)
                        rec_x1 = model.swinViT.mask_layer.get_masked_gt(rec_x1)
                    loss, contrastive_loss, rec_loss = loss_function(contrastive1_x, contrastive1_y, rec_x1, val_inputs)
                loss_val.append(loss.item())
                loss_val_recon.append(rec_loss.item())
                print("Validation step:{}, Loss:{:.4f}, Recloss:{:.4f}, Conloss:{:.4f}".format(step,
                                    loss.item(), rec_loss.item(), loss.item() - rec_loss.item()))

        return np.mean(loss_val), np.mean(loss_val_recon)


    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="/home/qlc/train_log", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=2.0, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=2.0, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")

    # masklayer
    parser.add_argument("--dynamic_masking", action="store_true")
    parser.add_argument("--hierarchical_masking", default=0., type=float)
    parser.add_argument("--basic_mask_ratio", default=0.5, type=float)
    parser.add_argument("--drop_ratio", default=0.3, type=float)
    parser.add_argument("--scale", default=0.3, type=float)

    parser.add_argument("--alpha1", default=0.005, type=float)
    parser.add_argument("--alpha2", default=1, type=float)
    parser.add_argument("--lambd", default=0.005, type=int)

    parser.add_argument("--datasets")
    parser.add_argument("--json_list", default="./jsons/brats21_folds.json", type=str, help="dataset json file")
    parser.add_argument("--fold", default=0, type=int, help="data fold")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")

    parser.add_argument("--invis_patches", action="store_true", help="calculate loss on masked patches")

    args = parser.parse_args()

    # Basic settings
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:3"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    global_step = 0
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # Setting output dir
    args.logdir = os.path.join(args.logdir, f'gmm_{args.basic_mask_ratio}_{args.scale}'
                                            f'_{args.drop_ratio}_{args.hierarchical_masking}_{args.invis_patches}')

    os.makedirs(args.logdir, exist_ok=True)
    if args.rank == 0:
        writer_dir = os.path.join(args.logdir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        writer = SummaryWriter(writer_dir)
    else:
        writer = None

    # Build model
    model = SSLHead(args)
    model.cuda()

    optimizer = None
    # Define optimizer and scheduler
    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    # Load checkpoint
    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"])
        global_step = global_step
        model.optimizer = model_dict["optimizer"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print("=> loaded optimizer checkpoint")        
        if "scheduler" in model_dict.keys():
            scheduler = scheduler.load_state_dict(model_dict['scheduler'])
            scheduler.step(epoch=global_step)

    # Define Loss
    loss_function = Loss(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # Get dataloader
    train_loader, test_loader = get_loader(datasets=args.datasets,
                                           datalist_json=args.json_list,
                                           fold=args.fold,
                                           batch_size=args.batch_size,
                                           num_works=8,
                                           args=args)

    # Start training
    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler)

    save_ckp(model, "final_ckpt.pt", optimizer=optimizer, scheduler=scheduler, global_step=global_step)


if __name__ == "__main__":
    main()
