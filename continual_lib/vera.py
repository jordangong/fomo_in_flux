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
                                self.adapter_dict[mod_layer] = VeRA_Conv2d(
                                    mod,
                                    self.rank,
                                    self.scale,
                                    *weight.shape[:3],
                                    name=mod_layer,
                                    device=device,
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
                                self.adapter_dict[mod_layer] = VeRA_Linear(
                                    mod,
                                    self.rank,
                                    self.scale,
                                    mod.out_features,
                                    mod.in_features,
                                    name=mod_layer,
                                    device=device,
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
                                # We only adapt Key & Value Heads via VeRA_KV as recommended in LAE (https://arxiv.org/abs/2303.10070).
                                # This helps when the expect adaptation shift is smaller.
                                self.adapter_dict[mod_layer] = VeRA_QKVlinear(
                                    mod,
                                    self.rank,
                                    self.scale,
                                    mod.in_features,
                                    mod.in_features,
                                    name=mod_layer,
                                    kv_only=self.kv_only,
                                    device=device,
                                )
                                mod.forward = self.adapter_dict[mod_layer].forward
                                self.to_optimize.extend(
                                    self.adapter_dict[mod_layer].to_optimize
                                )

                            # Apply VeRA Modules to Multihead Attention Layers in OAI/OPEN-CLIP models.
                            if (
                                mod_layer[-4:] == "attn"
                                and args.experiment.backbone.name
                                in backbones.clip_models + backbones.openclip_models
                            ):
                                print(f"Applied MHA Adapter to {mod_layer}.")
                                self.adapter_dict[mod_layer] = VeRA_MHA(
                                    mod,
                                    self.rank,
                                    self.scale,
                                    name=mod_layer,
                                    kv_only=self.kv_only,
                                    device=device,
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


class VeRA_Conv2d(torch.nn.Module):
    ### Note: we don't fix the varying sizes of random A and B matrices here, since this module only has a fixed set of kernel dims, in_channels, and out_channels in the ViT-B-32 arch

    ### init random shared frozen matrices
    vera_random_A = None
    vera_random_B = None

    def __init__(
        self,
        base_module,
        rank,
        scale,
        out_channels,
        in_channels,
        kernel_size,
        name=None,
        device='cuda',
    ):
        super().__init__()
        self.base_module = base_module
        self.rank = rank

        ### create shared random frozen matrices
        if self.__class__.vera_random_A is None or self.__class__.vera_random_B is None:
            self.__class__.create_shared_matrices(rank, in_channels, out_channels, kernel_size, device)

        ### create trainable vectors (init taken from sec 3.3 of https://arxiv.org/abs/2310.11454)
        self.vera_d = torch.nn.Parameter(
            self.base_module.weight.new_zeros((self.rank * kernel_size,))
        )
        self.vera_b = torch.nn.Parameter(
            0.1 * self.base_module.weight.new_ones((out_channels * kernel_size,))
        )
        self.to_optimize = [{"params": self.vera_d}, {"params": self.vera_b}]
        self.scale = scale
        self.assigned_name = name

    @classmethod
    def create_shared_matrices(cls, rank, in_channels, out_channels, kernel_size, device):
        # Initialize the matrices only if they have not been initialized yet
        if cls.vera_random_A is None or cls.vera_random_B is None:
            cls.vera_random_A = torch.nn.Parameter(torch.zeros((rank * kernel_size, in_channels * kernel_size)))
            cls.vera_random_B = torch.nn.Parameter(torch.zeros((out_channels * kernel_size, rank * kernel_size)))

            torch.nn.init.kaiming_uniform_(cls.vera_random_A, a=np.sqrt(rank))
            torch.nn.init.kaiming_uniform_(cls.vera_random_B, a=np.sqrt(rank))

            cls.vera_random_A.requires_grad = False
            cls.vera_random_B.requires_grad = False

            cls.vera_random_A = cls.vera_random_A.to(device)
            cls.vera_random_B = cls.vera_random_B.to(device)

    def forward(self, *input):

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device

        # Move the adapter weights and base module weights to the input device
        vera_b = self.vera_b.to(device)
        vera_random_B = self.vera_random_B.to(device)
        vera_d = self.vera_d.to(device)
        vera_random_A = self.vera_random_A.to(device)

        weight = self.scale * (torch.diag(vera_b) @ vera_random_B @ torch.diag(vera_d) @ vera_random_A).view(
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

class VeRA_Linear(torch.nn.Module):

    ### init random shared frozen matrices
    vera_random_As = None
    vera_random_Bs = None

    def should_add_new_random_mat(self, feat_dim_in, feat_dim_out):
        avail_dims = sorted(list(self.__class__.vera_random_As.keys()))
        if feat_dim_in not in avail_dims:
            return True

        avail_dims = sorted(list(self.__class__.vera_random_Bs.keys()))
        if feat_dim_out not in avail_dims:
            return True

        return False

    def __init__(self, base_module, rank, scale, out_features, in_features, name=None, device='cuda'):
        super().__init__()
        self.base_module = base_module
        self.rank = rank

        self.in_features = in_features
        self.out_features = out_features

        ### create shared random frozen matrices
        if self.__class__.vera_random_As is None or self.__class__.vera_random_Bs is None:
            self.__class__.create_shared_matrices(rank, in_features, out_features, device)
        elif self.should_add_new_random_mat(in_features, out_features):
            self.__class__.create_shared_matrices(rank, in_features, out_features, device)

        ### create trainable vectors (init taken from sec 3.3 of https://arxiv.org/abs/2310.11454)
        self.vera_d = torch.nn.Parameter(
            self.base_module.weight.new_zeros((self.rank,))
        )
        self.vera_b = torch.nn.Parameter(
            0.1 * self.base_module.weight.new_ones((out_features,))
        )
        self.to_optimize = [{"params": self.vera_d}, {"params": self.vera_b}]
        self.scale = scale
        self.assigned_name = name

    @classmethod
    def create_shared_matrices(cls, rank, in_features, out_features, device):
        # Index the matrices as a dict for easy retrieval based on in_features dim when doing the forward

        # Initialize the matrices as a dict only if they have not been initialized yet
        if cls.vera_random_As is None or cls.vera_random_Bs is None:

            cls.vera_random_As = {}
            cls.vera_random_Bs = {}

            cls.vera_random_As[in_features] = torch.nn.Parameter(torch.zeros((rank, in_features)))
            cls.vera_random_Bs[out_features] = torch.nn.Parameter(torch.zeros((out_features, rank)))

            torch.nn.init.kaiming_uniform_(cls.vera_random_As[in_features], a=np.sqrt(rank))
            torch.nn.init.kaiming_uniform_(cls.vera_random_Bs[out_features], a=np.sqrt(rank))

            cls.vera_random_As[in_features].requires_grad = False
            cls.vera_random_Bs[out_features].requires_grad = False

            cls.vera_random_As[in_features] = cls.vera_random_As[in_features].to(device)
            cls.vera_random_Bs[out_features] = cls.vera_random_Bs[out_features].to(device)

        # Add to existing dict if already initialized
        elif in_features not in cls.vera_random_As:

            cls.vera_random_As[in_features] = torch.nn.Parameter(torch.zeros((rank, in_features)))
            cls.vera_random_Bs[out_features] = torch.nn.Parameter(torch.zeros((out_features, rank)))

            torch.nn.init.kaiming_uniform_(cls.vera_random_As[in_features], a=np.sqrt(rank))
            torch.nn.init.kaiming_uniform_(cls.vera_random_Bs[out_features], a=np.sqrt(rank))

            cls.vera_random_As[in_features].requires_grad = False
            cls.vera_random_Bs[out_features].requires_grad = False

            cls.vera_random_As[in_features] = cls.vera_random_As[in_features].to(device)
            cls.vera_random_Bs[out_features] = cls.vera_random_Bs[out_features].to(device)

    def forward(self, *input, **kwargs):

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device

        # Move the adapter weights and base module weights to the input device
        vera_b = self.vera_b.to(device)
        vera_random_B = self.vera_random_Bs[self.out_features].to(device)
        vera_d = self.vera_d.to(device)
        vera_random_A = self.vera_random_As[self.in_features].to(device)

        weight = self.scale * (torch.diag(vera_b) @ vera_random_B @ torch.diag(vera_d) @ vera_random_A)

        # Move base module weights
        base_weight = self.base_module.weight.to(device)
        base_bias = self.base_module.bias.to(device) if self.base_module.bias is not None else None

        return torch.nn.functional.linear(
            *input, base_weight + weight, base_bias
        )

### Note: we dont test this since its not used in the vit-b-32 runs
class LoRA_QKVlinear(torch.nn.Module):
    ### Note: we don't fix the varying sizes of random A and B matrices here, since this module is not called in the main experiments

    ### init random shared frozen matrices
    vera_random_A = None
    vera_random_B = None

    def __init__(
        self,
        base_module,
        rank,
        scale,
        out_features,
        in_features,
        name=None,
        kv_only: bool = True,
        device='cuda',
    ):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.kv_only = kv_only
        mul = 2 if kv_only else 3

        ### create shared random frozen matrices
        if self.__class__.vera_random_A is None or self.__class__.vera_random_B is None:
            self.__class__.create_shared_matrices(rank, in_features, out_features, device)

        ### create trainable vectors (init taken from sec 3.3 of https://arxiv.org/abs/2310.11454)
        self.vera_d = torch.nn.ParameterList(
            [    
                torch.nn.Parameter(
                    self.base_module.weight.new_zeros((self.rank,))
                )
                for _ in range(mul)
            ]
        )

        self.vera_b = torch.nn.ParameterList(
            [    
                0.1 * torch.nn.Parameter(
                    self.base_module.weight.new_ones((out_features,))
                )
                for _ in range(mul)
            ]
        )
        self.to_optimize = [{"params": self.vera_d}, {"params": self.vera_b}]
        self.scale = scale
        self.assigned_name = name

    @classmethod
    def create_shared_matrices(cls, rank, in_features, out_features, device):
        # Initialize the matrices only if they have not been initialized yet
        if cls.vera_random_A is None or cls.vera_random_B is None:
            cls.vera_random_A = torch.nn.Parameter(torch.zeros((rank, in_features)))
            cls.vera_random_B = torch.nn.Parameter(torch.zeros((out_features, rank)))

            torch.nn.init.kaiming_uniform_(cls.vera_random_A, a=np.sqrt(rank))
            torch.nn.init.kaiming_uniform_(cls.vera_random_B, a=np.sqrt(rank))

            cls.vera_random_A.requires_grad = False
            cls.vera_random_B.requires_grad = False

            cls.vera_random_A = cls.vera_random_A.to(device)
            cls.vera_random_B = cls.vera_random_B.to(device)

    def forward(self, *input, **kwargs):
        weight = torch.cat(
            [self.scale * (torch.diag(b) @ self.vera_random_B @ torch.diag(d) @ self.vera_random_A) for b, d in zip(self.vera_b, self.vera_d)], dim=0
        )
        if self.kv_only:
            zeros = torch.zeros_like(self.base_module.weight)[: -weight.shape[0]]
            weight = torch.cat([zeros, weight], dim=0)
        return torch.nn.functional.linear(
            *input, self.base_module.weight + weight, self.base_module.bias
        )

class VeRA_MHA(torch.nn.Module):

    ### init random shared frozen matrices
    vera_random_As = None
    vera_random_Bs = None

    def should_add_new_random_mat(self, feat_dim):
        ### we add a new shared random matrix for every hidden dimension
        ### hence, we index it based on the keys of the random matrix dict
        avail_dims = sorted(list(self.__class__.vera_random_As.keys()))
        assert avail_dims == sorted(list(self.__class__.vera_random_Bs.keys())), 'vera_random_As and vera_random_Bs do not contain the same dims'
        if feat_dim not in avail_dims:
            return True
        return False

    def __init__(
        self, base_module, rank, scale, name: str = None, kv_only: bool = False, device: str = 'cuda',
    ):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.kv_only = kv_only
        mul = 2 if self.kv_only else 3
        self.mul = mul
        _, in_features = self.base_module.in_proj_weight.size()

        ### create shared random frozen matrices
        if self.__class__.vera_random_As is None or self.__class__.vera_random_Bs is None:
            self.__class__.create_shared_matrices(rank, in_features, device)
        elif self.should_add_new_random_mat(in_features):
            self.__class__.create_shared_matrices(rank, in_features, device)

        ### create trainable vectors (init taken from sec 3.3 of https://arxiv.org/abs/2310.11454)
        self.vera_d = torch.nn.ParameterList(
            [    
                torch.nn.Parameter(
                    self.base_module.in_proj_weight.new_zeros((self.rank,))
                )
                for _ in range(mul)
            ]
        )

        self.vera_b = torch.nn.ParameterList(
            [    
                0.1 * torch.nn.Parameter(
                    self.base_module.in_proj_weight.new_ones((in_features,))
                )
                for _ in range(mul)
            ]
        )
        self.to_optimize = [{"params": self.vera_d}, {"params": self.vera_b}]
        self.scale = scale
        self.assigned_name = name

    @classmethod
    def create_shared_matrices(cls, rank, in_features, device):
        # Index the matrices as a dict for easy retrieval based on in_features dim when doing the forward

        # Initialize the matrices as a dict only if they have not been initialized yet
        if cls.vera_random_As is None or cls.vera_random_Bs is None:

            cls.vera_random_As = {}
            cls.vera_random_Bs = {}

            cls.vera_random_As[in_features] = torch.nn.Parameter(torch.zeros((rank, in_features)))
            cls.vera_random_Bs[in_features] = torch.nn.Parameter(torch.zeros((in_features, rank)))

            torch.nn.init.kaiming_uniform_(cls.vera_random_As[in_features], a=np.sqrt(rank))
            torch.nn.init.kaiming_uniform_(cls.vera_random_Bs[in_features], a=np.sqrt(rank))

            cls.vera_random_As[in_features].requires_grad = False
            cls.vera_random_Bs[in_features].requires_grad = False

            cls.vera_random_As[in_features] = cls.vera_random_As[in_features].to(device)
            cls.vera_random_Bs[in_features] = cls.vera_random_Bs[in_features].to(device)

        # Add to existing dict if already initialized
        elif in_features not in cls.vera_random_As:

            cls.vera_random_As[in_features] = torch.nn.Parameter(torch.zeros((rank, in_features)))
            cls.vera_random_Bs[in_features] = torch.nn.Parameter(torch.zeros((in_features, rank)))

            torch.nn.init.kaiming_uniform_(cls.vera_random_As[in_features], a=np.sqrt(rank))
            torch.nn.init.kaiming_uniform_(cls.vera_random_Bs[in_features], a=np.sqrt(rank))

            cls.vera_random_As[in_features].requires_grad = False
            cls.vera_random_Bs[in_features].requires_grad = False

            cls.vera_random_As[in_features] = cls.vera_random_As[in_features].to(device)
            cls.vera_random_Bs[in_features] = cls.vera_random_Bs[in_features].to(device)

    def forward(self, *input, **kwargs):

        ### Fix for multi-gpu setup:
        ### since the base-modules are cloned
        ### from the graph, they are not 
        ### automatically moved to the right
        ### device --> we do this manually

        # Identify the device of the input tensor
        device = input[0].device

        # Move the adapter weights and base module weights to the input device
        vera_b = [param.to(device) for param in self.vera_b]
        vera_d = [param.to(device) for param in self.vera_d]
        vera_random_Bs = {k:v.to(device) for k,v in self.vera_random_Bs.items()}
        vera_random_As = {k:v.to(device) for k,v in self.vera_random_As.items()}

        in_proj_weight = torch.cat(
            [self.scale * (torch.diag(b) @ vera_random_Bs[b.shape[0]] @ torch.diag(d) @ vera_random_As[b.shape[0]]) for b, d in zip(vera_b, vera_d)], dim=0
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
