# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import numpy as np

import continual_lib
from continual_lib.merge import create_merge_instance, compute_dots


class Model(continual_lib.BaseContinualLearner):
    """
    This is the common merger utility.
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

        # Whether to include zero-shot checkpoint for merging or not?
        self.include_zero_shot_in_merge = self.args.continual.offline_merge.include_zero_shot_in_merge

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

        return {}

    def define_evaluation_weights(self, **kwargs):
        """Offline merge task evaluation weight init.
        
        For Offline merge, we unroll-merge over all stored task weights.
        """

        def linearly_increasing_list(n):
            # Generate a linearly increasing array from 1 to n
            line = np.linspace(1, n, n)
            # Normalize it so the sum is 1
            line /= line.sum()
            return line.tolist()

        def sqrt_scaling_list(n):
            # Generate a square root scaled array
            sqrt_values = np.array([np.sqrt(i) for i in range(1, n + 1)], dtype=float)
            # Normalize it so the sum is 1
            sqrt_values /= sqrt_values.sum()
            return sqrt_values.tolist()

        def quadratic_scaling_list(n):
            # Generate a quadratic scaled array
            quad_values = np.array([i**2 for i in range(1, n + 1)], dtype=float)
            # Normalize it so the sum is 1
            quad_values /= quad_values.sum()
            return quad_values.tolist()

        def cubic_scaling_list(n):
            # Generate a cubic scaled array
            quad_values = np.array([i**3 for i in range(1, n + 1)], dtype=float)
            # Normalize it so the sum is 1
            quad_values /= quad_values.sum()
            return quad_values.tolist()

        def fifth_power_scaling_list(n):
            # Generate a 5th power scaled array
            quad_values = np.array([i**5 for i in range(1, n + 1)], dtype=float)
            # Normalize it so the sum is 1
            quad_values /= quad_values.sum()
            return quad_values.tolist()

        def tenth_power_scaling_list(n):
            # Generate a 10th power scaled array
            quad_values = np.array([i**10 for i in range(1, n + 1)], dtype=float)
            # Normalize it so the sum is 1
            quad_values /= quad_values.sum()
            return quad_values.tolist()

        def exponentially_increasing_list(n, base=2):
            # Generate an exponentially increasing array as float
            exp_values = np.array([base**i for i in range(n)], dtype=float)
            # Normalize it so the sum is 1
            exp_values /= exp_values.sum()
            return exp_values.tolist()

        def logarithmic_scaling_list(n):
            # Generate a logarithmically increasing array as float
            log_values = np.array([np.log(i + 1) for i in range(1, n + 1)], dtype=float)
            # Normalize it so the sum is 1
            log_values /= log_values.sum()
            return log_values.tolist()

        for mode in ["backbone", "head"]:

            if self.include_zero_shot_in_merge:
                # if we include zero-shot checkpoint in merge
                if len(self.checkpoint_storage["running"][mode]) <= 1:
                    # if we only have zero-shot checkpoint, use just that
                    self.checkpoint_storage["eval"][mode] = copy.deepcopy(self.checkpoint_storage["running"][mode][-1])
                else:

                    # If merge method is interpolation, set the weights
                    if mode == "backbone":
                        if self.args.continual.offline_merge.backbone_merge.method == "interpolation":

                            if self.args.continual.offline_merge.interpolation_weighting == "linear":
                                weights_ = linearly_increasing_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "sqrt":
                                weights_ = sqrt_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "quadratic":
                                weights_ = quadratic_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "exp":
                                weights_ = exponentially_increasing_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "cubic":
                                weights_ = cubic_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "fifth":
                                weights_ = fifth_power_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "tenth":
                                weights_ = tenth_power_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "log":
                                weights_ = logarithmic_scaling_list(len(self.checkpoint_storage["running"][mode]))

                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_linear":
                                weights_ = linearly_increasing_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_sqrt":
                                weights_ = sqrt_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_quadratic":
                                weights_ = quadratic_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_exp":
                                weights_ = exponentially_increasing_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_cubic":
                                weights_ = cubic_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_fifth":
                                weights_ = fifth_power_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_tenth":
                                weights_ = tenth_power_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_log":
                                weights_ = logarithmic_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]

                            self.f_merge[mode].set_weight_coefficients(weight_coefficients=weights_)
                    else:
                        if self.args.continual.offline_merge.head_merge.method == "interpolation":

                            if self.args.continual.offline_merge.interpolation_weighting == "linear":
                                weights_ = linearly_increasing_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "sqrt":
                                weights_ = sqrt_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "quadratic":
                                weights_ = quadratic_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "exp":
                                weights_ = exponentially_increasing_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "cubic":
                                weights_ = cubic_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "fifth":
                                weights_ = fifth_power_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "tenth":
                                weights_ = tenth_power_scaling_list(len(self.checkpoint_storage["running"][mode]))
                            elif self.args.continual.offline_merge.interpolation_weighting == "log":
                                weights_ = logarithmic_scaling_list(len(self.checkpoint_storage["running"][mode]))

                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_linear":
                                weights_ = linearly_increasing_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_sqrt":
                                weights_ = sqrt_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_quadratic":
                                weights_ = quadratic_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_exp":
                                weights_ = exponentially_increasing_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_cubic":
                                weights_ = cubic_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_fifth":
                                weights_ = fifth_power_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_tenth":
                                weights_ = tenth_power_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_log":
                                weights_ = logarithmic_scaling_list(len(self.checkpoint_storage["running"][mode]))[::-1]

                            self.f_merge[mode].set_weight_coefficients(weight_coefficients=weights_)


                    # if we have multiple task checkpoints, merge all of them independently, including zero-shot checkpoint
                    self.checkpoint_storage["eval"][mode] = copy.deepcopy(self.f_merge[mode](
                        state_dicts=[copy.deepcopy(x) for x in self.checkpoint_storage["running"][mode]],
                        zero_shot=copy.deepcopy(self.checkpoint_storage["init"][mode]),
                    ))
            else:

                # if we do not include zero-shot checkpoint in merge
                if len(self.checkpoint_storage["running"][mode]) <= 2:
                    # if we don't have multiple tasks that have completed, use the latest checkpoint since theres nothing to merge.
                    self.checkpoint_storage["eval"][mode] = copy.deepcopy(self.checkpoint_storage["running"][mode][-1])
                else:

                    # If merge method is interpolation, set the weights
                    weights_ = None
                    if mode == "backbone":
                        if self.args.continual.offline_merge.backbone_merge.method == "interpolation":


                            if self.args.continual.offline_merge.interpolation_weighting == "linear":
                                weights_ = linearly_increasing_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "sqrt":
                                weights_ = sqrt_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "quadratic":
                                weights_ = quadratic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "exp":
                                weights_ = exponentially_increasing_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "cubic":
                                weights_ = cubic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "fifth":
                                weights_ = fifth_power_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "tenth":
                                weights_ = tenth_power_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "log":
                                weights_ = logarithmic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)

                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_linear":
                                weights_ = linearly_increasing_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_sqrt":
                                weights_ = sqrt_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_quadratic":
                                weights_ = quadratic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_exp":
                                weights_ = exponentially_increasing_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_cubic":
                                weights_ = cubic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_fifth":
                                weights_ = fifth_power_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_tenth":
                                weights_ = tenth_power_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_log":
                                weights_ = logarithmic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]

                            self.f_merge[mode].set_weight_coefficients(weight_coefficients=weights_)
                    else:
                        if self.args.continual.offline_merge.head_merge.method == "interpolation":


                            if self.args.continual.offline_merge.interpolation_weighting == "linear":
                                weights_ = linearly_increasing_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "sqrt":
                                weights_ = sqrt_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "quadratic":
                                weights_ = quadratic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "exp":
                                weights_ = exponentially_increasing_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "cubic":
                                weights_ = cubic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "fifth":
                                weights_ = fifth_power_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "tenth":
                                weights_ = tenth_power_scaling_list(len(self.checkpoint_storage["running"][mode])-1)
                            elif self.args.continual.offline_merge.interpolation_weighting == "log":
                                weights_ = logarithmic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)

                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_linear":
                                weights_ = linearly_increasing_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_sqrt":
                                weights_ = sqrt_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_quadratic":
                                weights_ = quadratic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_exp":
                                weights_ = exponentially_increasing_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_cubic":
                                weights_ = cubic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_fifth":
                                weights_ = fifth_power_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_tenth":
                                weights_ = tenth_power_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]
                            elif self.args.continual.offline_merge.interpolation_weighting == "reverse_log":
                                weights_ = logarithmic_scaling_list(len(self.checkpoint_storage["running"][mode])-1)[::-1]

                            self.f_merge[mode].set_weight_coefficients(weight_coefficients=weights_)

                    # if we have multiple task checkpoints, merge all of them independently
                    self.checkpoint_storage["eval"][mode] = copy.deepcopy(self.f_merge[mode](
                        state_dicts=[copy.deepcopy(x) for x in self.checkpoint_storage["running"][mode][1:]],
                        zero_shot=copy.deepcopy(self.checkpoint_storage["init"][mode]),
                    ))

    def define_training_weights(self, **kwargs):
        """Offline merge task training weight init.
        
        For Offline merge, we always initialize using the starting (zero-shot) weights.
        """
        for mode in ["backbone", "head"]:
            self.checkpoint_storage["train"][mode] = copy.deepcopy(self.checkpoint_storage["init"][mode])
        print('Set training weights to zero-shot')
