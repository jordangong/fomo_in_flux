import numpy as np
import torch
import tqdm

import continual_lib
from continual_lib.utils import buffer


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
        alpha,
        lambda_reg,
        max_imp_samples,
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.max_imp_samples = max_imp_samples

        self.penalty_checkpoint = self.backbone.module.get_params().data.clone()
        self.importances = None

    def penalty(self):
        if self.penalty_checkpoint is None or self.importances is None:
            penalty = torch.tensor(0.0).to(self.device)
        else:
            penalty = (
                self.importances
                * (self.backbone.module.get_params() - self.penalty_checkpoint) ** 2
            ).sum()
        return penalty

    def end_task(self, experiment, **kwargs):
        self.penalty_checkpoint = self.backbone.module.get_params().data.clone()

        num_iter = int(
            np.ceil(self.max_imp_samples / self.args.experiment.task.batch_size)
        )
        num_samples_seen = 0
        if num_iter >= len(experiment.current_task_train_loader):
            num_iter = len(experiment.current_task_train_loader)

        importances = torch.zeros_like(self.penalty_checkpoint)

        for batch_idx, data in enumerate(
            tqdm.tqdm(
                experiment.current_task_train_loader,
                desc="Computing FIM...",
                total=num_iter,
            )
        ):
            inputs, targets = data["inputs"].to(self.device), data["targets"].to(
                self.device
            )

            self.opt.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.backbone(inputs)
                # We reduce the norm size to avoid infinite gradients with mixed precision.
                loss = torch.norm(outputs, p="fro", dim=1).pow(2).mean() / 10000
            # loss.backward()
            self.scaler.scale(loss).backward()

            # importances += self.backbone.module.get_grads().abs()
            # We re-introduce the reductive factor from above, and account for mixed precision rescaling.
            importances += (
                self.backbone.module.get_grads().abs() / self.scaler.get_scale() * 10000
            )

            num_samples_seen += len(inputs)
            if batch_idx == num_iter - 1:
                break

        importances /= num_iter

        if self.importances is not None:
            self.importances = (
                1 - self.alpha
            ) * importances + self.alpha * self.importances
        else:
            self.importances = importances

    def observe(self, inputs, targets, **kwargs):
        self.opt.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = self.backbone(inputs)
            penalty = self.penalty()
            base_loss = self.loss(outputs, targets)
            loss = base_loss + self.lambda_reg * penalty

        self.gradient_update(loss)

        return loss.item()

    @property
    def checkpoint(self):
        return {
            "self": self.state_dict(),
            "importances": "none" if self.importances is None else self.importances,
            "penalty_checkpoint": (
                "none" if self.penalty_checkpoint is None else self.penalty_checkpoint
            ),
        }

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
        self.importances = (
            None
            if state_dict["importances"] == "none"
            else state_dict["importances"].to(self.device)
        )
        self.penalty_checkpoint = (
            None
            if state_dict["penalty_checkpoint"] == "none"
            else state_dict["penalty_checkpoint"]
        )
