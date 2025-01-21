import copy
from typing import List

import numpy as np
import torch

import continual_lib


class Model(continual_lib.BaseContinualLearner):
    """
    In the EMA-ONLINE-PAINT method, we store the following model weight copies:
    (1) theta_0, the zero-shot weights of the model
    (2) theta_u, the weights of the model that is iteratively fine-tuned.
    At each step, we update theta_u on the task data
    After each step, we perform the following merging step:
        current_backbone_weights = w * current_backbone_weights + (1-w) * theta_u
    """

    REQ_NON_AUG_INPUTS = False

    def __init__(
        self,
        args,
        backbone,
        head,
        loss,
        device,
        experiment,
        avg_weight,
        freeze_non_task_logits=None,
        **kwargs
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)
        self.weight_coefficient = avg_weight
        self.freeze_non_task_logits = freeze_non_task_logits
        self.global_tasks_seen = []
        self.task_mask = torch.ones(experiment.total_num_classes)
        self.seen_targets = []
        self.global_mask_idcs = []

        self.aux_device = device
        if device == "cuda" and torch.cuda.device_count() > 1:
            gpus = torch.cuda.device_count()
            aux_device_id = gpus // 2
            self.aux_device = f"cuda:{aux_device_id}"

        # keep zero-shot backbone weights on cpu to save GPU mem, can move them to the gpu mem when required
        self.zero_shot_model_dict_backbone = copy.deepcopy(
            {k: v.cpu() for k, v in self.backbone.state_dict().items()}
        )
        # currently storing two backbone model copies one on the cpu, one on the gpu
        self.paint_model_dict_backbone = copy.deepcopy(
            {k: v.to(self.aux_device) for k, v in self.backbone.state_dict().items()}
        )

        # keep zero-shot head weights on cpu to save GPU mem, can move them to the gpu mem when required
        self.zero_shot_model_dict_head = copy.deepcopy(
            {k: v.cpu() for k, v in self.head.state_dict().items()}
        )
        # currently storing two head model copies one on the cpu, one on the gpu
        self.paint_model_dict_head = copy.deepcopy(
            {k: v.to(self.aux_device) for k, v in self.head.state_dict().items()}
        )

    def observe(self, images, targets, **kwargs):
        # step through the update_model in each batch of a given task
        self.opt.zero_grad()

        global_task = kwargs["experiment"].global_task
        with torch.cuda.amp.autocast():
            # Get masking indices if needed.
            if self.freeze_non_task_logits is not None:
                if (
                    self.freeze_non_task_logits
                    and global_task not in self.global_tasks_seen
                ):
                    if global_task not in self.global_tasks_seen:
                        self.task_mask = torch.ones(kwargs["experiment"].total_num_classes)
                        warm_logit_idcs = kwargs[
                            "experiment"
                        ].give_class_indices_of_current_task_targets()
                        self.task_mask[warm_logit_idcs] = 0
                        self.global_tasks_seen.append(global_task)

            outputs = self.forward(images=images, **kwargs)
            # Mask unrelated logits if needed.
            if self.freeze_non_task_logits:
                outputs[:, torch.where(self.task_mask)[0]] = -float("inf")

            logit_scale = getattr(self.head.module.text_encoder, "logit_scale", 1.0)
            temp = 1.0 / logit_scale.exp()
            loss = self.loss(targets=targets, temperature=temp, **outputs, **kwargs)

        self.gradient_update(loss)

        # after each step, set the backbone weights and update_model weights to the update
        # from eqn at top of this script; do the weight merging on cpu to save GPU memory
        with torch.no_grad():
            ### update backbone
            self.paint_model_dict_backbone = self.average_weights(
                self.backbone.state_dict(), self.paint_model_dict_backbone
            )

            ### update head
            self.paint_model_dict_head = self.average_weights(
                self.head.state_dict(), self.paint_model_dict_head
            )

        return loss.item()

    def end_task(self, experiment, **kwargs):
        # Update base backbone to use interpolated weights for evaluation.
        self.backbone.load_state_dict(self.paint_model_dict_backbone)

        # Update base head to use interpolated weights for evaluation.
        self.head.load_state_dict(self.paint_model_dict_head)

    def average_weights(self, weight_dict1, weight_dict2):
        """
        Averages the weights of two weight dictionaries, `weight_dict1` and `weight_dict2`, based on the `weight_coefficient` attribute.

        Parameters:
        - weight_dict1 (dict): The first weight dictionary.
        - weight_dict2 (dict): The second weight dictionary.

        Returns:
        - dict: A dictionary with the averaged weights.

        Raises:
        - AssertionError: If the two dictionaries have unequal keys.

        Note:
        The averaging formula for each weight is:
        (1 - self.weight_coefficient) * weight from weight_dict1 + self.weight_coefficient * weight from weight_dict2
        """

        # ensure the weight merging is always done on the same device
        weight_dict1 = {k: v.to(self.aux_device) for k, v in weight_dict1.items()}
        weight_dict2 = {k: v.to(self.aux_device) for k, v in weight_dict2.items()}

        # ensure both the weight dicts have the same key set
        assert set(weight_dict1.keys()) == set(
            weight_dict2.keys()
        ), "Cannot merge weight dictionaries with unequal keys."
        theta = {
            key: (1 - self.weight_coefficient) * weight_dict1[key]
            + self.weight_coefficient * weight_dict2[key]
            for key in weight_dict1.keys()
        }
        return theta

    @property
    def checkpoint(self):
        return {"self": self.state_dict()}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
