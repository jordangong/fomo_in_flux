# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import tqdm

import continual_lib


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(
        self,
        args,
        backbone,
        head,
        loss,
        device,
        experiment,
        e_lambda,
        gamma,
        fim_avg,
        max_fim_samples,
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)
        self.logsoft = torch.nn.LogSoftmax(dim=1)

        self.penalty_checkpoint_backbone = None
        self.penalty_checkpoint_head = None

        self.fim_backbone = None
        self.fim_head = None

        self.gamma = gamma
        self.e_lambda = e_lambda
        self.fim_avg = fim_avg
        self.max_fim_samples = max_fim_samples

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
        ## just check for the backbone checkpoints since we do ops for backbone and head together anyway
        if self.penalty_checkpoint_backbone is None or self.fim_backbone is None:
            penalty_backbone = torch.tensor(0.0).cuda()
            penalty_head = torch.tensor(0.0).cuda()
        else:
            penalty_backbone = (
                (
                    self.fim_backbone
                    * (
                        (
                            self.get_backbone_params().cpu()
                            - self.penalty_checkpoint_backbone.cpu()
                        )
                        ** 2
                    )
                )
                .sum()
                .cuda()
            )

            penalty_head = (
                (
                    self.fim_head
                    * (
                        (
                            self.head.module.get_params().cpu()
                            - self.penalty_checkpoint_head.cpu()
                        )
                        ** 2
                    )
                )
                .sum()
                .cuda()
            )

        return penalty_backbone + penalty_head

    def end_task(self, experiment, **kwargs):

        ## do all offloading on cpu to save gpu memory
        self.penalty_checkpoint_head = (
            self.head.module.get_params().data.clone().cpu()
        )  ## currently tested only for the ClipTextHead head type which we use in our experiments
        self.penalty_checkpoint_backbone = self.get_backbone_params().data.clone().cpu()

        num_iter = int(
            np.ceil(self.max_fim_samples / self.args.experiment.task.batch_size)
        )

        max_fim_samples = self.max_fim_samples
        num_samples_seen = 0
        if num_iter >= len(experiment.current_train_loader):
            num_iter = len(experiment.current_train_loader)
        if max_fim_samples >= len(experiment.current_train_loader.dataset):
            max_fim_samples = len(experiment.current_train_loader.dataset)

        # currently on cpu
        fim_backbone = torch.zeros_like(self.penalty_checkpoint_backbone)
        fim_head = torch.zeros_like(self.penalty_checkpoint_head)

        for batch_idx, data in enumerate(
            tqdm.tqdm(
                experiment.current_train_loader,
                desc="Computing FIM...",
                total=num_iter,
            )
        ):
            images, targets = data["images"].cuda(), data["targets"].cuda()
            if "texts" in data:
                texts = data["texts"]
            else:
                texts = None

            self.opt.zero_grad()
            with torch.amp.autocast("cuda"):
                # Update get grads options.
                outputs = self.forward(
                    experiment=experiment, images=images, texts=texts, targets=targets
                )
                logit_scale = getattr(self.head.module.text_encoder, "logit_scale", 1.0)
                temp = 1.0 / logit_scale
                loss = self.loss(targets=targets, temperature=temp, **outputs, **kwargs)

            self.scaler.scale(loss).backward()

            # Because of mixed precision, we have to unscale the gradients before using them for our diagonal FIM approximation.
            fim_backbone += (
                self.get_backbone_grads().cpu() / self.scaler.get_scale()
            ) ** 2
            fim_head += (
                self.head.module.get_grads().cpu() / self.scaler.get_scale()
            ) ** 2

            num_samples_seen += len(images)
            if batch_idx == num_iter - 1:
                break

        fim_head /= num_iter
        fim_backbone /= num_iter

        if self.fim_backbone is None:
            self.fim_backbone = fim_backbone
        else:
            self.fim_backbone *= self.gamma
            self.fim_backbone += fim_backbone

        if self.fim_head is None:
            self.fim_head = fim_head
        else:
            self.fim_head *= self.gamma
            self.fim_head += fim_head

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
            # eq 3 in https://arxiv.org/abs/1612.00796
            loss = base_loss + self.e_lambda / 2 * penalty

        self.gradient_update(loss)

        return loss.item()

    @property
    def checkpoint(self):
        return {
            "self": self.state_dict(),
            "fim_head": "none" if self.fim_head is None else self.fim_head,
            "fim_backbone": "none" if self.fim_backbone is None else self.fim_backbone,
            "penalty_checkpoint_backbone": (
                "none"
                if self.penalty_checkpoint_backbone is None
                else self.penalty_checkpoint_backbone
            ),
            "penalty_checkpoint_head": (
                "none"
                if self.penalty_checkpoint_head is None
                else self.penalty_checkpoint_head
            ),
        }

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
        self.fim_head = (
            None if state_dict["fim_head"] == "none" else state_dict["fim_head"].cuda()
        )
        self.fim_backbone = (
            None
            if state_dict["fim_backbone"] == "none"
            else state_dict["fim_backbone"].cuda()
        )

        self.penalty_checkpoint_backbone = (
            None
            if state_dict["penalty_checkpoint_backbone"] == "none"
            else state_dict["penalty_checkpoint_backbone"]
        )

        self.penalty_checkpoint_head = (
            None
            if state_dict["penalty_checkpoint_head"] == "none"
            else state_dict["penalty_checkpoint_head"]
        )
