from typing import List

import numpy as np
import torch

import backbones
import continual_lib

### TODO: we should check both settings: continual lora tuning vs. continual tune-then-merging

class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(
        self,
        args,
        backbone,
        head,
        loss,
        device,
        scale: float = 1,
        rank: int = 5,
        backbone_block_idcs: List[int] = [0, 1, 2, 3, 4, 5],
        head_block_idcs: List[int] = [0, 1, 2, 3, 4, 5],
        kv_only: bool = True,
        tune_logit_scale = False,
        **kwargs,
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)

        self.scale = scale
        self.rank = rank
        self.backbone_block_idcs = backbone_block_idcs
        self.head_block_idcs = head_block_idcs
        self.kv_only = kv_only

        ### First, freeze all backbone and head params
        for n, w in self.backbone.named_parameters():
            w.requires_grad = False

        for n, w in self.head.named_parameters():
            if tune_logit_scale:
                if 'logit_scale' in n:
                    print(n, w.shape)
                    w.requires_grad = True
                else:
                    w.requires_grad = False
            else:
                w.requires_grad = False

        ### Define adapter parameters.
        self.adapter_dict = {}
        self.to_optimize = []

        # Get backbone & head model blocks where adapters should be attached to.
        backbone_adapter_blocks = None
        if not self.freeze_features:
            backbone_adapter_blocks = backbones.backbone_separators[
                args.experiment.backbone.name
            ]
            if len(self.backbone_block_idcs):
                backbone_adapter_blocks = [
                    backbone_adapter_blocks[i] for i in self.backbone_block_idcs
                ]

        head_adapter_blocks = None
        if not self.freeze_head and args.experiment.backbone.head == "default":
            head_adapter_blocks = backbones.head_separators[
                args.experiment.backbone.name
            ]
            if len(self.head_block_idcs):
                head_adapter_blocks = [
                    head_adapter_blocks[i] for i in self.head_block_idcs
                ]

        blocks = [backbone_adapter_blocks, head_adapter_blocks]
        modules = [self.backbone, self.head]

        # Attach respective adapters to backbone and/or head.
        for module_blocks, module in zip(blocks, modules):
            if module_blocks is not None:
                for layer, weight in module.module.named_parameters():
                    mod = module.module
                    if not len(module_blocks) or any(
                        [x + "." in layer for x in module_blocks]
                    ):
                        if "bias" not in layer:
                            mod_layer = ".".join(layer.split(".")[:-1])
                            for name in mod_layer.split("."):
                                mod = mod._modules[name]

                            if (
                                isinstance(mod, torch.nn.Conv2d)
                                and "patch_embed" not in mod_layer
                            ):
                                print(f"Applied Conv2D-Adapter to {mod_layer}.")
                                # We don't adapt patch-embeddings for ViT-style architectures.
                                self.adapter_dict[mod_layer] = LoRA_Conv2d(
                                    mod,
                                    self.rank,
                                    self.scale,
                                    *weight.shape[:3],
                                    name=mod_layer,
                                )
                                mod.forward = self.adapter_dict[mod_layer].forward
                                self.to_optimize.extend(
                                    self.adapter_dict[mod_layer].to_optimize
                                )

                            if (
                                isinstance(mod, torch.nn.Linear)
                                and "head" not in mod_layer
                                and "fc" not in mod_layer
                            ):
                                print(f"Applied Linear Adapter to {mod_layer}.")
                                self.adapter_dict[mod_layer] = LoRA_Linear(
                                    mod,
                                    self.rank,
                                    self.scale,
                                    mod.out_features,
                                    mod.in_features,
                                    name=mod_layer,
                                )
                                mod.forward = self.adapter_dict[mod_layer].forward
                                self.to_optimize.extend(
                                    self.adapter_dict[mod_layer].to_optimize
                                )

                            # This is the standard QKV-linear projection using in TIMM-style VITs.
                            if (
                                isinstance(mod, torch.nn.Linear)
                                and "qkv" in mod_layer
                                and args.experiment.backbone.name
                                in backbones.timm_models
                            ):
                                print(f"Applied QKV Adapter to {mod_layer}.")
                                # We only adapt Key & Value Heads via LoRA_KV as recommended in LAE (https://arxiv.org/abs/2303.10070).
                                # This helps when the expect adaptation shift is smaller.
                                self.adapter_dict[mod_layer] = LoRA_QKVlinear(
                                    mod,
                                    self.rank,
                                    self.scale,
                                    mod.in_features,
                                    mod.in_features,
                                    name=mod_layer,
                                    kv_only=self.kv_only,
                                )
                                mod.forward = self.adapter_dict[mod_layer].forward
                                self.to_optimize.extend(
                                    self.adapter_dict[mod_layer].to_optimize
                                )

                            # Apply LoRA Modules to Multihead Attention Layers in OAI/OPEN-CLIP models.
                            if (
                                mod_layer[-4:] == "attn"
                                and args.experiment.backbone.name
                                in backbones.clip_models + backbones.openclip_models
                            ):
                                print(f"Applied MHA Adapter to {mod_layer}.")
                                self.adapter_dict[mod_layer] = LoRA_MHA(
                                    mod,
                                    self.rank,
                                    self.scale,
                                    name=mod_layer,
                                    kv_only=self.kv_only,
                                )
                                # Include KV-only option as well.
                                mod.forward = self.adapter_dict[mod_layer].forward
                                self.to_optimize.extend(
                                    self.adapter_dict[mod_layer].to_optimize
                                )

        # If the head is not meant to be frozen and no head adapters are used, we optimize the full head.
        if not self.freeze_head and head_adapter_blocks is None:
            self.to_optimize.append({"params": self.head.parameters()})

        if tune_logit_scale:
            self.to_optimize.append({"params": self.head.module.text_encoder.logit_scale})

    def observe(self, images, targets, **kwargs):
        self.opt.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = self.forward(images=images.cuda(), **kwargs)
            logit_scale = getattr(self.head.module.text_encoder, "logit_scale", 1.0)
            temp = 1.0 / logit_scale.exp()
            loss = self.loss(targets=targets.cuda(), temperature=temp, **outputs, **kwargs)

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

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device        

        # Move the adapter weights and base module weights to the input device
        lora_A = self.lora_A.to(device)
        lora_B = self.lora_B.to(device)

        weight = self.scale * (lora_B @ lora_A).view(
            self.base_module.weight.shape
        )

        # Move base module weights
        base_weight = self.base_module.weight.to(device)
        base_bias = self.base_module.bias.to(device) if self.base_module.bias is not None else None

        return torch.nn.functional.conv2d(
            *input,
            base_weight + weight,
            base_bias,
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

    def forward(self, *input, **kwargs):

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device        

        # Move the adapter weights and base module weights to the input device
        lora_A = self.lora_A.to(device)
        lora_B = self.lora_B.to(device)

        weight = self.scale * (lora_B @ lora_A)

        # Move base module weights
        base_weight = self.base_module.weight.to(device)
        base_bias = self.base_module.bias.to(device) if self.base_module.bias is not None else None

        return torch.nn.functional.linear(
            *input, base_weight + weight, base_bias
        )

### Note: have not tested this impl since its not used for the main vit-b-32 runs
class LoRA_QKVlinear(torch.nn.Module):
    def __init__(
        self,
        base_module,
        rank,
        scale,
        out_features,
        in_features,
        name=None,
        kv_only: bool = True,
    ):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.kv_only = kv_only
        mul = 2 if kv_only else 3
        self.lora_A = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    self.base_module.weight.new_zeros((self.rank, in_features))
                )
                for _ in range(mul)
            ]
        )
        self.lora_B = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    self.base_module.weight.new_zeros((out_features, self.rank))
                )
                for _ in range(mul)
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

    def forward(self, *input, **kwargs):
        weight = torch.cat(
            [self.scale * B @ A for A, B in zip(self.lora_A, self.lora_B)], dim=0
        )
        if self.kv_only:
            zeros = torch.zeros_like(self.base_module.weight)[: -weight.shape[0]]
            weight = torch.cat([zeros, weight], dim=0)
        return torch.nn.functional.linear(
            *input, self.base_module.weight + weight, self.base_module.bias
        )


class LoRA_MHA(torch.nn.Module):
    def __init__(
        self, base_module, rank, scale, name: str == None, kv_only: bool = False
    ):
        # design choices: we only adapt in_proj_weight and not all proj matrices
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.kv_only = kv_only
        mul = 2 if self.kv_only else 3
        self.mul = mul
        _, in_features = self.base_module.in_proj_weight.size()
        self.lora_A = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    self.base_module.in_proj_weight.new_zeros((self.rank, in_features))
                )
                for _ in range(mul)
            ]
        )
        self.lora_B = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    self.base_module.in_proj_weight.new_zeros((in_features, self.rank))
                )
                for _ in range(mul)
            ]
        )
        self.to_optimize = [{"params": self.lora_A}, {"params": self.lora_B}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.mul):
            torch.nn.init.kaiming_uniform_(self.lora_A[i], a=np.sqrt(self.rank))
            torch.nn.init.zeros_(self.lora_B[i])

    def forward(self, *input, **kwargs):

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device        

        # Move the adapter weights and base module weights to the input device
        lora_A = [param.to(device) for param in self.lora_A]
        lora_B = [param.to(device) for param in self.lora_B]

        in_proj_weight = torch.cat(
            [self.scale * B @ A for A, B in zip(lora_A, lora_B)], dim=0
        )
        if self.kv_only:
            zeros = torch.zeros_like(self.base_module.in_proj_weight)[
                : -in_proj_weight.shape[0]
            ]
            in_proj_weight = torch.cat([zeros, in_proj_weight], dim=0)

        # Move the base module weights to the correct device
        base_module_weights = self.base_module.in_proj_weight.to(device)
        base_module_bias = self.base_module.in_proj_bias.to(device)
        out_proj_weight = self.base_module.out_proj.weight.to(device)
        out_proj_bias = self.base_module.out_proj.bias.to(device)

        return torch.nn.functional.multi_head_attention_forward(
            *input,
            self.base_module.embed_dim,
            self.base_module.num_heads,
            base_module_weights + in_proj_weight,
            base_module_bias,
            self.base_module.bias_k,
            self.base_module.bias_v,
            self.base_module.add_zero_attn,
            self.base_module.dropout,
            out_proj_weight,
            out_proj_bias,
            training=self.training,
            **kwargs,
        )
