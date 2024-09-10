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
import math
import sys
from typing import Iterable, Optional

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from util.model_ema_sched import ModelEma_sched
import torch.nn.functional as F

def loss_ftn(output1, output2):
    output1 = F.normalize(output1, dim=-1, p=2)
    output2 = F.normalize(output2, dim=-1, p=2)
    return 2 - 2 * (output1 * output2).sum(dim=-1)

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    model_ema: Optional[ModelEma_sched] = None,
                    log_writer=None, args=None, print_freq=20):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    len_data_loader = len(data_loader)
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = [im.to(device, non_blocking=True) for im in samples]

        with torch.cuda.amp.autocast():
            mae_loss, student_output, _ = model(samples[0], samples[1], mask_ratio=args.mask_ratio)
            _, teacher_output, _ = model_ema.ema(samples[0], samples[1], mask_ratio=args.mask_ratio, ema=True)

            student_outs = student_output.chunk(2)
            teacher_outs = teacher_output.detach().chunk(2)

            bc_loss = 0
            n_loss_terms = 0
            for iq, q in enumerate(teacher_outs):
                for v in range(len(student_outs)):
                    if v == iq:
                        continue
                    loss = loss_ftn(student_outs[v], q)
                    bc_loss += loss.mean()
                    n_loss_terms += 1
            bc_loss /= n_loss_terms

        mae_loss_value = mae_loss.item()
        bc_loss_value = bc_loss.item()
        total_loss_value = args.w_mae * mae_loss_value + args.w_bc * bc_loss_value

        if not math.isfinite(total_loss_value):
            print("Loss is {}, stopping training".format(total_loss_value))
            sys.exit(1)

        mae_loss /= accum_iter
        bc_loss /= accum_iter
        total_loss = args.w_mae * mae_loss + args.w_bc * bc_loss
        grad_norm = loss_scaler(total_loss, optimizer, parameters=model.parameters(), # ------
                                update_grad=(data_iter_step + 1) % accum_iter == 0)

        if model_ema is not None:
            it = len_data_loader * epoch + data_iter_step
            if (data_iter_step + 1) % accum_iter == 0:
                model_ema.update(model, it)

        loss_scale_value = loss_scaler.state_dict()["scale"]
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=total_loss_value)
        metric_logger.update(mae_loss=mae_loss_value)
        metric_logger.update(bc_loss=bc_loss_value) # ------
        metric_logger.update(loss_scale=loss_scale_value)
        if model_ema is not None:
            if (data_iter_step + 1) % accum_iter == 0:
                metric_logger.update(ema_decay=model_ema.decay[it])

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        weight_decay_value = 0
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 \
                and log_writer.logger_type() == 'tensorboard':
            log_writer.update(loss=total_loss_value, head="loss")
            log_writer.update(mae_loss=mae_loss_value, head="mae_loss")
            log_writer.update(bc_loss=bc_loss_value, head="bc_loss") # ------
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            if model_ema is not None:
                if (data_iter_step + 1) % accum_iter == 0:
                    log_writer.update(ema_decay=model_ema.decay[it], head="opt")

            log_writer.set_step()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
