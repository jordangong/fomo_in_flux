import copy
from typing import List

import numpy as np
import torch

import continual_lib
from continual_lib.merge import compute_dots


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

        # Init EMA model dicts
        self.ema_model_dict = {
            "backbone": copy.deepcopy({k: v.to(self.aux_device) for k, v in self.backbone.state_dict().items()}),
            "head": copy.deepcopy({k: v.to(self.aux_device) for k, v in self.head.state_dict().items()})
        }

    def observe(self, images, targets, **kwargs):
        # step through the update_model in each batch of a given task
        self.opt.zero_grad()

        global_task = kwargs["experiment"].global_task
        with torch.amp.autocast("cuda"):
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
        # from eqn at top of this script
        with torch.no_grad():
            ### update backbone
            self.ema_model_dict["backbone"] = self.average_weights(
                self.backbone.state_dict(), self.ema_model_dict["backbone"]
            )

            ### update head
            self.ema_model_dict["head"] = self.average_weights(
                self.head.state_dict(), self.ema_model_dict["head"]
            )

        return loss.item()

    def end_task(self, experiment, **kwargs):
        # at the end of each task, merge the backbone and head weights using the specified merge technique
        with torch.no_grad():
            base_state_dicts = {
                "backbone": {k: v.cpu() for k, v in self.backbone.state_dict().items()},
                "head": {k: v.cpu() for k, v in self.head.state_dict().items()},
            }

            # update backbone
            dots = {}
            for mode in ["backbone", "head"]:
                # We store post-training evaluated weights here.
                self.checkpoint_storage["running"][mode].append(copy.deepcopy(base_state_dicts[mode]))
                # Compute respective dot products.
                dots[mode] = compute_dots(
                    base_state_dicts[mode], self.checkpoint_storage["train"][mode]
                )
        return {
            **{f"dot-prods.backbone.{k}": v for k, v in dots["backbone"].items()},
            **{f"dot-prods.head.{k}": v for k, v in dots["head"].items()},
        }

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

    def define_evaluation_weights(self, **kwargs):
        for mode in ["backbone", "head"]:
            self.checkpoint_storage["eval"][mode] = copy.deepcopy(self.ema_model_dict[mode])

    def define_training_weights(self, **kwargs):
        for mode in ["backbone", "head"]:
            self.checkpoint_storage["train"][mode] = copy.deepcopy(self.ema_model_dict[mode])
