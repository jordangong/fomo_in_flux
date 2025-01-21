import copy
from typing import List

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
        prompt_length: int = 5,
        prompt_top_k: int = 5,
        prompt_pool_size: int = 10,
        learnable_keys: bool = True,
        prompt_align_weight: float = 0.1,
        batchwise_prompts: bool = False,
        output_mode: str = "average",
        **kwargs,
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)

        self.top_k = prompt_top_k
        self.prompt_align_weight = prompt_align_weight

        self.storage = {}

        # Attach prompt-modifier to backbone.
        from IPython import embed

        embed()
        patch_module = backbones.patch_modules[self.args.experiment.backbone.name]
        assert (
            patch_module is not None
        ), f"Backbone model [{self.args.experiment.backbone.name}] has no patch-generating module to attach prompt-modifier too!"

        # TODO: Update feature dim / prompt dim, handle bs x dim x h x w output instead of bs x n x dim.

        self.prompt_modifier = PromptModifier(
            self.storage,
            args,
            device,
            backbone,
            prompt_pool_size,
            prompt_length,
            self.backbone_feature_dim,
            prompt_top_k,
            learnable_keys,
            batchwise_prompts,
        ).to(device)
        getattr(self.backbone.module, patch_module).register_forward_hook(
            self.prompt_modifier
        )

        # We average / modify the final output embeddings associated with the length of the input prompt before feeding it to the classifier head.
        if hasattr(backbone.module, "attn_pool"):
            print(
                "L2P: Backbone model has attn_pool module to attach altered output modifier too!"
            )
            self.attn_modifier = AttnModifier(
                self.prompt_modifier.total_prompt_length, output_mode
            )
            self.backbone.module.attn_pool = self.attn_modifier

        # Define parameters to be optimized (keys & values in prompt pool, classifier head)
        self.to_optimize = [
            {"params": self.prompt_modifier.prompts},
            {"params": self.prompt_modifier.prompt_keys},
        ]

        # TODO: Include head-based L2P here as well if not self.freeze_head & args.experiment.backbone.head == 'default'
        # custom_head_prompts = False
        # if not self.freeze_head & args.experiment.backbone.head == 'default':
        #     pass

        if not self.freeze_head:
            # if not self.freeze_head and not custom_head_prompts:
            self.to_optimize.append({"params": self.head.parameters()})

    # TODO Here
    def attach_modifiers(self, model, args, device, prompt_modifier_param_dict):
        # Has to be able to attach to both patch_embed and language input elements!
        self.prompt_modifier = PromptModifier(
            self.storage, args, device, model, **prompt_modifier_param_dict
        )
        self.backbone.module.patch_embed.register_forward_hook(self.prompt_modifier)

    def observe(self, images, targets, **kwargs):
        self.opt.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = self.forward(images=images, **kwargs)
            loss = self.loss(targets=targets, **outputs, **kwargs)
            loss -= self.prompt_align_weight * self.storage["key_query_sim"]

        self.gradient_update(loss)

        return loss.item()

    @property
    def checkpoint(self):
        return {"self": self.state_dict()}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])


class PromptModifier(torch.nn.Module):
    def __init__(
        self,
        storage,
        args,
        device,
        backbone,
        prompt_pool_size,
        prompt_length,
        prompt_dim,
        prompt_top_k: int = 5,
        learnable_keys: bool = True,
        batchwise_prompts: bool = False,
    ):
        """Adjust Input-Prompts for Transformer-style models

        Hooks onto the Patch-Embed function of any ViT Transformer Model.
        """
        super(PromptModifier, self).__init__()
        self.device = device
        self.prompt_top_k = prompt_top_k
        self.prompt_length = prompt_length
        self.prompt_dim = prompt_dim
        self.learnable_keys = learnable_keys
        self.batchwise_prompts = batchwise_prompts

        self.storage = storage
        self.storage["key_query_sim"] = 0

        self.query_model = copy.deepcopy(backbone)
        self.query_model_hooks = {}
        self.query_model_hook_handle = continual_lib.utils.hook_default_features(
            args, self.query_model.module, self.query_model_hooks
        )
        _ = self.query_model.eval()

        # Adjust positional embedding lengths in-place of base backbone model to allow for prompt-extensions.
        self.total_prompt_length = prompt_length * prompt_top_k

        # Augment the positional encodings to account for input prompts.
        if hasattr(backbone.module, "pos_embed"):
            ref_device = backbone.module.pos_embed.device
            backbone.module.pos_embed = torch.nn.Parameter(
                torch.cat(
                    [
                        torch.nn.Parameter(
                            torch.randn(1, self.total_prompt_length, prompt_dim).to(
                                ref_device
                            )
                            * 0.02
                        ),
                        backbone.module.pos_embed,
                    ],
                    dim=1,
                )
            )

        prompt_pool_shape = (prompt_pool_size, prompt_length, prompt_dim)
        self.prompts = torch.nn.Parameter(torch.randn(prompt_pool_shape))
        _ = torch.nn.init.uniform_(self.prompts, -1, 1)

        prompt_key_shape = (prompt_pool_size, prompt_dim)
        if self.learnable_keys:
            self.prompt_keys = torch.nn.Parameter(torch.randn(prompt_key_shape))
            _ = torch.nn.init.uniform_(self.prompt_keys, -1, 1)
        else:
            self.prompt_keys = torch.mean(self.prompts, dim=1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(
            torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))
        )
        return x * x_inv_norm

    def __call__(self, module, input, output):
        # Compute query using frozen model version.
        with torch.no_grad():
            _ = self.query_model(input[0])
            prompt_queries = self.query_model_hooks["features"]
            prompt_queries = self.l2_normalize(prompt_queries, dim=-1)

        prompt_keys = self.l2_normalize(self.prompt_keys, dim=-1)
        qk_sims = torch.matmul(prompt_queries, prompt_keys.T)
        _, idx = torch.topk(qk_sims, k=self.prompt_top_k, dim=1)

        if self.batchwise_prompts:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            prompt_pool_size = self.prompts.shape[0]
            if prompt_id.shape[0] < prompt_pool_size:
                prompt_id = torch.cat(
                    [
                        prompt_id,
                        torch.full(
                            (prompt_pool_size - prompt_id.shape[0],),
                            torch.min(idx.flatten()),
                            device=prompt_id.device,
                        ),
                    ]
                )
                id_counts = torch.cat(
                    [
                        id_counts,
                        torch.full(
                            (prompt_pool_size - id_counts.shape[0],),
                            0,
                            device=id_counts.device,
                        ),
                    ]
                )
            _, major_idx = torch.topk(id_counts, k=self.prompt_top_k)
            major_prompt_id = prompt_id[major_idx]
            idx = major_prompt_id.expand(output.shape[0], -1)  # B, top_k

        batched_prompt = self.prompts[idx].reshape(
            len(output), self.prompt_top_k * self.prompt_length, self.prompt_dim
        )

        batched_prompt_keys = prompt_keys[idx]
        kq_sims = batched_prompt_keys * prompt_queries.unsqueeze(1)
        self.storage["key_query_sim"] = torch.sum(kq_sims) / prompt_queries.shape[0]

        from IPython import embed

        embed()

        return torch.cat([batched_prompt, output], dim=1)


class AttnModifier:
    def __init__(self, total_prompt_length, mode="average"):
        self.total_prompt_length = total_prompt_length
        self.mode = mode

    def __call__(self, input):
        if self.mode == "average":
            # x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            return input[:, : self.total_prompt_length].mean(dim=1)
        elif self.mode == "cls":
            return input[:, 0]
