import torch
from torch.optim import Optimizer
import math


def init_opt(encoder, iterations_per_epoch, start_lr, ref_lr, ref_mom, nesterov, warmup, batches, weight_decay=1e-6, final_lr=0.0):
    param_groups = [
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' not in n) and ('bn' not in n))},
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' in n) or ('bn' in n)),
         'LARS_exclude': True,
         'weight_decay': 0}
    ]
    optimizer = SGD(param_groups, weight_decay=weight_decay, momentum=ref_mom, nesterov=nesterov, lr=ref_lr)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup*iterations_per_epoch, start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr, T_max=batches)
    optimizer = LARS(optimizer, trust_coefficient=0.001)
    return encoder, optimizer, scheduler


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        super(WarmupCosineSchedule, self).__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
            return new_lr / self.ref_lr

        # -- progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.T_max - self.warmup_steps))
        new_lr = max(self.final_lr,
                     self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
        return new_lr / self.ref_lr


class LARS(torch.optim.Optimizer):

    def __init__(self, optimizer, trust_coefficient=0.02, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            # stats = AverageMeter()
            weight_decays = []
            for group in self.optim.param_groups:

                # -- takes weight decay control from wrapped optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)

                # -- user wants to exclude this parameter group from LARS
                #    adaptation
                if ('LARS_exclude' in group) and group['LARS_exclude']:
                    continue
                group['weight_decay'] = 0

                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        # stats.update(adaptive_lr)
                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # -- return weight decay control to wrapped optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

        # return stats


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
class SGD(Optimizer):

    def __init__(self, params, lr, momentum=0, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if nesterov and (momentum == 0.0):
            raise ValueError(f'Nesterov needs momentum > 0')

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov)
        
        super(SGD, self).__init__(params, defaults)
        
        
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                d_p.mul_(-group['lr'])

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone().detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p)

        return None
