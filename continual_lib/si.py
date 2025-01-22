# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import continual_lib


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(self, args, backbone, head, loss, device, experiment, c, xi):
        super(Model, self).__init__(args, backbone, head, loss, device)

        self.backbone_checkpoint = self.get_backbone_params().data.clone().cpu()
        self.head_checkpoint = self.head.module.get_params().data.clone().cpu()

        self.big_omega_backbone = None
        self.small_omega_backbone = 0

        self.big_omega_head = None
        self.small_omega_head = 0

        self.c = c
        self.xi = xi

    #### get backbone params directly here since sometimes the backbone is a direct open_clip backbone
    #### and it can be tricky to directly create a get_params() fn like for the heads.
    #### the better fix would of-course be to get it directly in the backbone code, but for now leaving it as is
    def get_backbone_params(self):
        params = []
        for pp in list(self.backbone.module.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    #### get grads for backbone directly here for the same reason as above
    def get_backbone_grads(self):
        return torch.cat(self.get_backbone_grads_list())

    def get_backbone_grads_list(self):
        grads = []
        for pp in list(self.backbone.module.parameters()):
            if pp.grad is not None:
                grads.append(pp.grad.view(-1))
            else:
                grads.append(torch.zeros_like(pp).view(-1))
        return grads

    def penalty(self):
        if self.big_omega_backbone is None:
            return torch.tensor(0.0).cuda()
        else:
            penalty_backbone = (
                (
                    self.big_omega_backbone.cpu()
                    * (
                        (self.get_backbone_params().cpu() - self.backbone_checkpoint)
                        ** 2
                    )
                )
                .sum()
                .cuda()
            )

            penalty_head = (
                (
                    self.big_omega_head.cpu()
                    * (
                        (self.head.module.get_params().cpu() - self.head_checkpoint)
                        ** 2
                    )
                )
                .sum()
                .cuda()
            )

            return penalty_backbone + penalty_head

    def end_task(self, experiment, **kwargs):
        # big omega calculation step
        if self.big_omega_backbone is None:
            self.big_omega_backbone = torch.zeros_like(
                self.get_backbone_params()
            ).cuda()
            self.big_omega_head = torch.zeros_like(self.head.module.get_params()).cuda()

        self.big_omega_backbone += self.small_omega_backbone / (
            (self.get_backbone_params().data.cuda() - self.backbone_checkpoint.cuda())
            ** 2
            + self.xi
        )

        self.big_omega_head += self.small_omega_head / (
            (self.head.module.get_params().data.cuda() - self.head_checkpoint.cuda())
            ** 2
            + self.xi
        )

        # store parameters checkpoint and reset small_omega
        self.backbone_checkpoint = self.get_backbone_params().data.clone().cpu()
        self.head_checkpoint = self.head.module.get_params().data.clone().cpu()
        self.small_omega_head = 0
        self.small_omega_backbone = 0

    def observe(self, images, targets, **kwargs):
        self.opt.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = self.forward(images=images, **kwargs)
            logit_scale = getattr(self.head.module.text_encoder, "logit_scale", 1.0)
            temp = 1.0 / logit_scale.exp()
            base_loss = self.loss(
                targets=targets, temperature=temp, **outputs, **kwargs
            )

            penalty = self.penalty()
            # eq 4 in https://arxiv.org/abs/1703.04200
            loss = base_loss + self.c * penalty

        self.scaler.scale(loss).backward()

        grad_clip_norm = self.args.experiment.optimizer.clip_grad_norm
        if grad_clip_norm > 0:
            self.scaler.unscale_(self.opt)
            gradient_norm_clipper = lambda params: (
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
                if grad_clip_norm > 0
                else lambda x: x
            )
            # We clip potential non-backbone parameters that are optimized.
            optim_parameters = [x["params"] for x in self.opt.param_groups]
            optim_parameters = [x for y in optim_parameters for x in y]
            _ = gradient_norm_clipper(optim_parameters)
            # Now we make sure to also clip all relevant base backbone parameters.
            _ = gradient_norm_clipper(self.backbone.parameters())

        if not any(
            torch.isnan(self.head.module.get_grads().data).detach().cpu().numpy()
        ):
            self.small_omega_head += (
                self.opt.param_groups[0]["lr"] * self.head.module.get_grads().data ** 2
            )
        if not any(torch.isnan(self.get_backbone_grads().data).detach().cpu().numpy()):
            self.small_omega_backbone += (
                self.opt.param_groups[0]["lr"] * self.get_backbone_grads().data ** 2
            )

        self.scaler.step(self.opt)
        self.scaler.update()

        return loss.item()

    @property
    def checkpoint(self):
        return {
            "self": self.state_dict(),
            "backbone_checkpoint": self.backbone_checkpoint.cpu(),
            "head_checkpoint": self.head_checkpoint.cpu(),
            "big_omega_backbone": (
                "none"
                if self.big_omega_backbone is None
                else self.big_omega_backbone.cpu()
            ),
            "big_omega_head": (
                "none" if self.big_omega_head is None else self.big_omega_head.cpu()
            ),
            "small_omega_backbone": self.small_omega_backbone,
            "small_omega_head": self.small_omega_head,
        }

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
        self.backbone_checkpoint = state_dict["backbone_checkpoint"].cuda()
        self.head_checkpoint = state_dict["head_checkpoint"].cuda()

        self.big_omega_backbone = (
            None
            if state_dict["big_omega_backbone"] == "none"
            else state_dict["big_omega_backbone"].cuda()
        )

        self.big_omega_head = (
            None
            if state_dict["big_omega_head"] == "none"
            else state_dict["big_omega_head"].cuda()
        )

        self.small_omega_backbone = state_dict["small_omega_backbone"]
        self.small_omega_head = state_dict["small_omega_head"]
