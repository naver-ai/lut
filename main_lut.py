"""
This code was originally obtained from:
https://github.com/facebookresearch/mae
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.optim.optim_factory as optim_factory

import util.logger as logger
from util.model_saver import ModelSaver
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.model_ema_sched import ModelEma_sched
from util.augs import GaussianBlur, Solarization, cosine_scheduler

import models
from engines.engine_lut import train_one_epoch

from functools import partial

import warnings

warnings.filterwarnings("ignore", message="Argument interpolation should be of type InterpolationMode instead of int. "
                                          "Please, use InterpolationMode enum.")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--save_periods', nargs='+', default=['last2', 'every_200_epochs'],
                        help='periods for save')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Extra parameters
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequency for print operation')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=False)

    # EMA parameters
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Ditillation parameters
    parser.add_argument('--depth_head', default=2, type=int)
    parser.add_argument('--depth_pred', default=2, type=int)
    parser.add_argument('--head_mlp_dim', default=4096, type=int,
                        help='embedding dimension for a projector and a predictor')
    parser.add_argument('--head_norm_layer', default=None, type=str,
                        help='type of norm layers in a projector and a predictor [None /BN / LN] ')
    parser.add_argument('--head_act_layer', default='ReLU', type=str,
                        help='type of norm layers in a projector and a predictor [ReLU / GELU] ')
    
    # loss weights
    parser.add_argument('--w_mae', type=float, default=1.0,
                        help='reconstruction pretask weight')
    parser.add_argument('--w_b', type=float, default=0.1,
                        help='Broader Contextualization loss weight')
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    saver = None
    if args.output_dir:
        saver = ModelSaver(checkpoint_dir=args.output_dir, target='local',
                           periods=args.save_periods)

    transform_train = TwoGlobalViewAugmentation()
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = logger.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    norm_list = {
        'BN': nn.BatchNorm1d,
        'LN': partial(nn.LayerNorm, eps=1e-6),
    }
    act_list = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
    }

    model = models.__dict__[args.model](depth_head=args.depth_head, depth_pred=args.depth_pred, head_mlp_dim=args.head_mlp_dim,
                                        head_norm_layer = norm_list[args.head_norm_layer] if args.head_norm_layer else args.head_norm_layer,
                                        head_act_layer = act_list[args.head_act_layer],
                                        norm_pix_loss=args.norm_pix_loss)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    model_ema = None
    if args.model_ema:
        ema_decay_schedule = cosine_scheduler(args.model_ema_decay, 1,
                                              args.epochs, len(data_loader_train))
        model_ema = ModelEma_sched(
            model,
            decay=ema_decay_schedule,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        # args.lr = args.blr * eff_batch_size / 256
        args.lr = args.blr * math.sqrt(eff_batch_size) / 4

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("base lr: %.2e" % (args.lr * 4 / math.sqrt(eff_batch_size)))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.auto_load_last_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None and log_writer.logger_type() == 'tensorboard':
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            model_ema,
            log_writer=log_writer,
            args=args,
            print_freq=args.print_freq
        )

        save_flag = misc.is_main_process()
        if args.output_dir and save_flag:
            save_dict = misc.save_dict(args=args, model=model,
                                       model_without_ddp=model_without_ddp,
                                       optimizer=optimizer,
                                       loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
            saver.save(step=epoch,
                       num_steps=args.epochs,
                       state=save_dict,
                       summary={'epoch': '%d/%d' % (
                           epoch + 1, args.epochs), **train_stats, })

        log_stats = {**{f'pretrain/{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if log_writer is not None:
            if log_writer.logger_type() == 'tensorboard':
                if args.output_dir and misc.is_main_process():
                    log_writer.flush()
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


class TwoGlobalViewAugmentation(object):
    def __init__(self):

        three_augment = transforms.Compose([transforms.RandomChoice([transforms.RandomGrayscale(p=1.0),
                                                  Solarization(p=1.0),
                                                  GaussianBlur(p=1.0)])])
        simple_resized_crop = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=3),
            transforms.RandomCrop(args.input_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            simple_resized_crop,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            simple_resized_crop,
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            three_augment,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        return crops


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
