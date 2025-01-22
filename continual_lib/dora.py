from typing import List

import numpy as np
import torch

import backbones
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
                                self.adapter_dict[mod_layer] = DoRA_Conv2d(
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
                                self.adapter_dict[mod_layer] = DoRA_Linear(
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
                                # We only adapt Key & Value Heads via DoRA_KV as recommended in LAE (https://arxiv.org/abs/2303.10070).
                                # This helps when the expect adaptation shift is smaller.
                                self.adapter_dict[mod_layer] = DoRA_QKVlinear(
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

                            # Apply DoRA Modules to Multihead Attention Layers in OAI/OPEN-CLIP models.
                            if (
                                mod_layer[-4:] == "attn"
                                and args.experiment.backbone.name
                                in backbones.clip_models + backbones.openclip_models
                            ):
                                print(f"Applied MHA Adapter to {mod_layer}.")
                                self.adapter_dict[mod_layer] = DoRA_MHA(
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


class DoRA_Conv2d(torch.nn.Module):
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

        # magnitude vector: dim k x 1
        self.weight_m_wdecomp = torch.nn.Parameter(
            self.base_module.weight.new_zeros((out_channels * kernel_size, 1))
        )

        # direction matrices: A -> dim r x k, B -> dim d x r
        self.dora_A = torch.nn.Parameter(
            self.base_module.weight.new_zeros((self.rank * kernel_size, in_channels * kernel_size))
        )
        self.dora_B = torch.nn.Parameter(
            self.base_module.weight.new_zeros((out_channels * kernel_size, self.rank * kernel_size))
        )
        self.to_optimize = [{"params": self.dora_A}, {"params": self.dora_B}, {"params": self.weight_m_wdecomp}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        # initialize the norm-scale with the original weight's norm and the dora A and B with lora init, see 
        # (1) https://github.com/AnswerDotAI/fsdp_qlora/blob/1a9fddfb660b85e46bdf18c3fcf0d599628c9c6c/tests/test_dora.py#L64C38-L64C67
        # (2) https://github.com/AnswerDotAI/fsdp_qlora/blob/main/scripts/dora.py
        # (3) sec 4.1, eq 5 of paper: https://arxiv.org/abs/2402.09353
        self.weight_m_wdecomp.data = self.base_module.weight.norm(p=2, dim=1)
        torch.nn.init.kaiming_uniform_(self.dora_A, a=np.sqrt(self.rank))
        torch.nn.init.zeros_(self.dora_B)

    def forward(self, *input):

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device

        # Move the adapter weights and base module weights to the input device
        dora_B = self.dora_B.to(device)
        dora_A = self.dora_A.to(device)
        weight_m_wdecomp = self.weight_m_wdecomp.to(device)

        # Update base module weights
        base_weight = self.base_module.weight.to(device)
        base_bias = self.base_module.bias.to(device) if self.base_module.bias is not None else None

        # numerator of update: v + \Delta v
        new_weight_v = base_weight + (dora_B @ dora_A) * self.scale

        # denominator of update: || numerator || = || v + \Delta v ||
        # we detach this from the computational graph to not backprop gradients for this op, as suggested in sec 4.3 of the paper: https://arxiv.org/abs/2402.09353
        norm_scale = weight_m_wdecomp.view(-1) / (torch.linalg.norm(new_weight_v, dim=1)).detach()

        weight = norm_scale.unsqueeze(1) * (base_weight + (dora_B @ dora_A) * self.scale)
        weight = weight.view(self.base_module.weight.shape)

        return torch.nn.functional.conv2d(
            *input,
            weight,
            base_bias,
            self.base_module.stride,
            self.base_module.padding,
            self.base_module.dilation,
            self.base_module.groups,
        )

class DoRA_Linear(torch.nn.Module):
    def __init__(self, base_module, rank, scale, out_features, in_features, name=None):
        super().__init__()
        self.base_module = base_module
        self.rank = rank

        # magnitude vector: dim k x 1
        self.weight_m_wdecomp = torch.nn.Parameter(
            self.base_module.weight.new_zeros((out_features, 1))
        )

        # direction matrices: A -> dim r x k, B -> dim d x r
        self.dora_A = torch.nn.Parameter(
            self.base_module.weight.new_zeros((self.rank, in_features))
        )
        self.dora_B = torch.nn.Parameter(
            self.base_module.weight.new_zeros((out_features, self.rank))
        )
        self.to_optimize = [{"params": self.dora_A}, {"params": self.dora_B}, {"params": self.weight_m_wdecomp}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        # initialize the norm-scale with the original weight's norm and the dora A and B with lora init, see 
        # (1) https://github.com/AnswerDotAI/fsdp_qlora/blob/1a9fddfb660b85e46bdf18c3fcf0d599628c9c6c/tests/test_dora.py#L64C38-L64C67
        # (2) https://github.com/AnswerDotAI/fsdp_qlora/blob/main/scripts/dora.py
        # (3) sec 4.1, eq 5 of paper: https://arxiv.org/abs/2402.09353
        self.weight_m_wdecomp.data = self.base_module.weight.norm(p=2, dim=1)
        torch.nn.init.kaiming_uniform_(self.dora_A, a=np.sqrt(self.rank))
        torch.nn.init.zeros_(self.dora_B)

    def forward(self, *input, **kwargs):

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device

        # Move the adapter weights and base module weights to the input device
        dora_B = self.dora_B.to(device)
        dora_A = self.dora_A.to(device)
        weight_m_wdecomp = self.weight_m_wdecomp.to(device)

        # Update base module weights
        base_weight = self.base_module.weight.to(device)
        base_bias = self.base_module.bias.to(device) if self.base_module.bias is not None else None

        # numerator of update: v + \Delta v
        new_weight_v = base_weight + (dora_B @ dora_A) * self.scale

        # denominator of update: || numerator || = || v + \Delta v ||
        # we detach this from the computational graph to not backprop gradients for this op, as suggested in sec 4.3 of the paper: https://arxiv.org/abs/2402.09353
        norm_scale = weight_m_wdecomp.view(-1) / (torch.linalg.norm(new_weight_v, dim=1)).detach()

        weight = norm_scale.unsqueeze(-1) * (base_weight + (dora_B @ dora_A) * self.scale)
        return torch.nn.functional.linear(
            *input, 
            weight, 
            base_bias,
        )

### This is not updated/tested since we don't use this in any of the ViT-B-32 exps
class DoRA_QKVlinear(torch.nn.Module):
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


class DoRA_MHA(torch.nn.Module):
    def __init__(
        self, base_module, rank, scale, name: str == None, kv_only: bool = False
    ):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.kv_only = kv_only
        mul = 2 if self.kv_only else 3
        self.mul = mul
        _, in_features = self.base_module.in_proj_weight.size()

        ### TODO: for now we only allow tuning all qkv proj matrices
        assert self.kv_only == False, 'For now, we only allow tuning all qkv proj matrices'

        # magnitude vector: dim k x 1
        # here: since the output vector size is going to be of dim k * self.mul, we do not need to have self.mul 
        # independent magnitude params, it is sufficient to have just a single magnitude param of size k * self.mul
        self.weight_m_wdecomp = torch.nn.Parameter(
            self.base_module.in_proj_weight.new_zeros((in_features * self.mul, 1))
        )

        # direction matrices: A -> dim r x k, B -> dim d x r
        self.dora_A = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    self.base_module.in_proj_weight.new_zeros((self.rank, in_features))
                )
                for _ in range(mul)
            ]
        )
        
        self.dora_B = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    self.base_module.in_proj_weight.new_zeros((in_features, self.rank))
                )
                for _ in range(mul)
            ]
        )
        self.to_optimize = [{"params": self.dora_A}, {"params": self.dora_B}, {"params": self.weight_m_wdecomp}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        # initialize the norm-scale with the original weight's norm and the dora A and B with lora init, see 
        # (1) https://github.com/AnswerDotAI/fsdp_qlora/blob/1a9fddfb660b85e46bdf18c3fcf0d599628c9c6c/tests/test_dora.py#L64C38-L64C67
        # (2) https://github.com/AnswerDotAI/fsdp_qlora/blob/main/scripts/dora.py
        # (3) sec 4.1, eq 5 of paper: https://arxiv.org/abs/2402.09353
        self.weight_m_wdecomp.data = self.base_module.in_proj_weight.norm(p=2, dim=1)
        for i in range(self.mul):
            torch.nn.init.kaiming_uniform_(self.dora_A[i], a=np.sqrt(self.rank))
            torch.nn.init.zeros_(self.dora_B[i])

    def forward(self, *input, **kwargs):

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device

        # Move the adapter weights and base module weights to the input device
        dora_B = [param.to(device) for param in self.dora_B]
        dora_A = [param.to(device) for param in self.dora_A]
        weight_m_wdecomp = self.weight_m_wdecomp.to(device)

        # \Delta v
        add_numerator_ = torch.cat(
            [
                self.scale * (dB_ @ dA_)
                for dB_, dA_ in zip(dora_B, dora_A)
            ]
        )

        # Update base module weights
        base_weight = self.base_module.in_proj_weight.to(device)
        base_bias = self.base_module.in_proj_bias.to(device)
        out_weight = self.base_module.out_proj.weight.to(device)
        out_bias = self.base_module.out_proj.bias.to(device)

        # numerator of update: v + \Delta v
        new_weight_v = base_weight + add_numerator_

        # denominator of update: || numerator || = || v + \Delta v ||
        # we detach this from the computational graph to not backprop gradients for this op, as suggested in sec 4.3 of the paper: https://arxiv.org/abs/2402.09353
        norm_scale = weight_m_wdecomp.view(-1) / (torch.linalg.norm(new_weight_v, dim=1)).detach()

        add_num_for_grads_ = torch.cat(
            [
                self.scale * (dB_ @ dA_)
                for dB_, dA_ in zip(dora_B, dora_A)
            ]
        )

        weight = norm_scale.unsqueeze(1) * (base_weight + add_num_for_grads_)

        return torch.nn.functional.multi_head_attention_forward(
            *input,
            self.base_module.embed_dim,
            self.base_module.num_heads,
            weight,
            base_bias,
            self.base_module.bias_k,
            self.base_module.bias_v,
            self.base_module.add_zero_attn,
            self.base_module.dropout,
            out_weight,
            out_bias,
            training=self.training,
            **kwargs,
        )
