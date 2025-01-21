# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List

import torch

import continual_lib
from continual_lib.merge import create_merge_instance, compute_dots

### TODO (vishaal): we haven't yet checked if the copy.deepcopy issue is fixed for this script

class Model(continual_lib.BaseContinualLearner):
    """ZS-PAINT method for continual learning.

    This method stores two model weight copies:
    1. theta_0: the zero-shot weights of the model
    2. theta_u: the weights of the model that is iteratively fine-tuned

    In each task, theta_u is updated on the task data but initialized from the zero-shot weights.
    At the end of each task, a merging step is performed using a specified merge method.

    Attributes:
        REQ_NON_AUG_INPUTS (bool): Flag indicating if non-augmented inputs are required.
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
        backbone_merge,
        head_merge,
        freeze_non_task_logits=None,
        **kwargs,
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)
        self.freeze_non_task_logits = freeze_non_task_logits
        self.global_tasks_seen = []
        self.task_mask = torch.ones(experiment.total_num_classes)
        self.seen_targets = []
        self.global_mask_idcs = []

        # Initialize merge methods
        self.f_merge = {
            "backbone": create_merge_instance(backbone_merge),
            "head": create_merge_instance(head_merge),
        }

    def observe(self, images, targets, **kwargs):
        """Perform a single training step.

        Args:
            images: Input images.
            targets: Target labels.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The loss value for this step.
        """
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
                        self.task_mask = torch.ones(
                            kwargs["experiment"].total_num_classes
                        )
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
        return loss.item()

    def prepare_for_training(
        self,
        **kwargs
    ):
        """Initialize model weights for training on a given task.
        
        In ZS-Paint, we simply always initialize from pretraining weights.
        """
        self.backbone.load_state_dict(self.put_gpu(self.checkpoint_storage["init"]["backbone"]))
        self.head.load_state_dict(self.put_gpu(self.checkpoint_storage["init"]["head"]))

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

    def unrolled_merge(self, f_merge, checkpoint_list):
        with torch.no_grad():
            running_ckpt = checkpoint_list[0]
            for chkpt in checkpoint_list[1:]:
                running_ckpt = f_merge([chkpt, running_ckpt], zero_shot=checkpoint_list[0])
            return running_ckpt

    def define_evaluation_weights(self, **kwargs):
        """ZS Paint task training weight init.
        
        For ZS-Paint, we unroll-merge over all stored task weights.
        """
        for mode in ["backbone", "head"]:
            self.checkpoint_storage["eval"][mode] = copy.deepcopy(self.unrolled_merge(
                self.f_merge[mode], self.checkpoint_storage["running"][mode]
            ))

    def define_training_weights(self, **kwargs):
        """ZS Paint task training weight init.
        
        For ZS-Paint, we always initialize using the starting weights.
        """
        for mode in ["backbone", "head"]:
            self.checkpoint_storage["train"][mode] = copy.deepcopy(self.checkpoint_storage["init"][mode])
