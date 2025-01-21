import copy
from typing import List

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
        scale: float = 1,
        rank: int = 5,
        layers: List[int] = [],
        freeze_non_task_logits: bool = False,
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)

        self.scale = scale
        self.rank = rank
        self.layers = layers

        ####
        self.freeze_non_task_logits = freeze_non_task_logits
        self.global_tasks_seen = []
        self.task_mask = torch.ones(experiment.total_num_classes)
        self.seen_targets = []
        self.global_mask_idcs = []

        ### Define adapter parameters.
        self.adapter_dict = {}
        self.to_optimize = []

        for layer, weight in self.backbone.module.named_parameters():
            mod = self.backbone.module
            if "bias" not in layer:
                mod_layer = ".".join(layer.split(".")[:-1])
                for name in mod_layer.split("."):
                    mod = mod._modules[name]

                if isinstance(mod, torch.nn.Conv2d) and "patch_embed" not in mod_layer:
                    # We don't adapt patch-embeddings for ViT-style architectures.
                    self.adapter_dict[mod_layer] = LoRA_Conv2d(
                        mod, self.rank, self.scale, *weight.shape[:3], name=mod_layer
                    )
                    mod.forward = self.adapter_dict[mod_layer].forward
                    self.to_optimize.extend(self.adapter_dict[mod_layer].to_optimize)

                if (
                    isinstance(mod, torch.nn.Linear)
                    and "head" not in mod_layer
                    and "fc" not in mod_layer
                ):
                    if "proj" in mod_layer:
                        pass
                        # self.adapter_dict[mod_layer] = LoRA_Linear(mod, self.rank, self.scale, mod.out_features, mod.in_features, name=mod_layer)
                        # mod.forward = self.adapter_dict[mod_layer].forward
                        # self.to_optimize.extend(self.adapter_dict[mod_layer].to_optimize)
                    elif "qkv" in mod_layer:
                        # We only adapt Key & Value Heads via LoRA_KV.
                        self.adapter_dict[mod_layer] = LoRA_KV(
                            mod,
                            self.rank,
                            self.scale,
                            mod.in_features,
                            mod.in_features,
                            name=mod_layer,
                        )
                        mod.forward = self.adapter_dict[mod_layer].forward
                        self.to_optimize.extend(
                            self.adapter_dict[mod_layer].to_optimize
                        )

        # Define parameters to be optimized (keys & values in prompt pool, classifier head)
        if self.backbone.module.head is not None:
            self.to_optimize.append({"params": self.backbone.module.head.parameters()})

    def observe(self, inputs, targets, **kwargs):
        self.opt.zero_grad()
        global_task = kwargs["experiment"].global_task
        with torch.cuda.amp.autocast():
            if (
                self.freeze_non_task_logits
                and global_task not in self.global_tasks_seen
            ):
                if global_task not in self.global_tasks_seen:
                    total_num_classes = kwargs["experiment"].total_num_classes
                    self.task_mask = torch.ones(total_num_classes)
                    warm_logit_idcs = kwargs[
                        "experiment"
                    ].give_class_indices_of_current_task_targets()
                    self.task_mask[warm_logit_idcs] = 0
                    # self.seen_targets.extend(list(warm_logit_idcs))
                    # self.seen_targets = sorted(list(set(self.seen_targets)))
                    # self.global_mask_idcs = sorted(list(set(range(total_num_classes)) - set(self.seen_targets)))
                    self.global_tasks_seen.append(global_task)

            outputs = self.backbone(inputs)
            if self.freeze_non_task_logits:
                outputs[:, torch.where(self.task_mask)[0]] = -float("inf")
            loss = self.loss(outputs, targets)

        self.gradient_update(loss)

        return loss.item()

    @property
    def checkpoint(self):
        return {"self": self.state_dict()}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])


class LoRA_Conv2d(torch.nn.Module):
    def __init__(
        self,
        base_module,
        rank,
        scale,
        out_channels,
        in_channels,
        kernel_size,
        name=None,
    ):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.lora_A = torch.nn.Parameter(
            self.base_module.weight.new_zeros(
                (self.rank * kernel_size, in_channels * kernel_size)
            )
        )
        self.lora_B = torch.nn.Parameter(
            self.base_module.weight.new_zeros(
                (out_channels * kernel_size, self.rank * kernel_size)
            )
        )
        self.to_optimize = [{"params": self.lora_A}, {"params": self.lora_B}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(self.rank))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, *input):
        weight = self.scale * (self.lora_B @ self.lora_A).view(
            self.base_module.weight.shape
        )
        return torch.nn.functional.conv2d(
            *input,
            self.base_module.weight + weight,
            self.base_module.bias,
            self.base_module.stride,
            self.base_module.padding,
            self.base_module.dilation,
            self.base_module.groups,
        )


class LoRA_Linear(torch.nn.Module):
    def __init__(self, base_module, rank, scale, out_features, in_features, name=None):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.lora_A = torch.nn.Parameter(
            self.base_module.weight.new_zeros((self.rank, in_features))
        )
        self.lora_B = torch.nn.Parameter(
            self.base_module.weight.new_zeros((out_features, self.rank))
        )
        self.to_optimize = [{"params": self.lora_A}, {"params": self.lora_B}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(self.rank))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, *input):
        weight = self.scale * (self.lora_B @ self.lora_A)
        return torch.nn.functional.linear(
            *input, self.base_module.weight + weight, self.base_module.bias
        )


class LoRA_KV(torch.nn.Module):
    def __init__(self, base_module, rank, scale, out_features, in_features, name=None):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.lora_A = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    self.base_module.weight.new_zeros((self.rank, in_features))
                )
                for _ in range(2)
            ]
        )
        self.lora_B = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    self.base_module.weight.new_zeros((out_features, self.rank))
                )
                for _ in range(2)
            ]
        )
        self.to_optimize = [{"params": self.lora_A}, {"params": self.lora_B}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(2):
            torch.nn.init.kaiming_uniform_(self.lora_A[i], a=np.sqrt(self.rank))
            torch.nn.init.zeros_(self.lora_B[i])

    def forward(self, *input):
        weight = torch.cat(
            [self.scale * B @ A for A, B in zip(self.lora_A, self.lora_B)], dim=0
        )
        zeros = torch.zeros_like(self.base_module.weight)[: -weight.shape[0]]
        weight = torch.cat([zeros, weight], dim=0)
        return torch.nn.functional.linear(
            *input, self.base_module.weight + weight, self.base_module.bias
        )
