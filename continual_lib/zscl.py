import copy
import json
from typing import List

import tqdm

import continual_lib

import torch
import torch.nn.functional as F


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
            temp,
            we_freq,
            l2,
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)
        self.distill_temp = temp
        self.we_freq = we_freq
        self.l2 = l2

        self.zs_backbone = copy.deepcopy(backbone)
        self.aux_device = device
        if device == "cuda" and torch.cuda.device_count() > 1:
            gpus = torch.cuda.device_count()
            aux_device_id = gpus // 2
            aux_dp_ids = list(range(aux_device_id, gpus))
            self.aux_device = f"cuda:{aux_device_id}"
            zs_backbone = self.zs_backbone.module.to(self.aux_device)
            self.zs_backbone = torch.nn.DataParallel(zs_backbone, device_ids=aux_dp_ids)

        self.iteration = 0
        self.we_step = 0
        self.we_backbone_weights = None
        self.we_head_weights = None

        ref_texts = json.load(open("data_lib/00_info/cc3m_val_captions.json"))
        ref_text_embed = []
        ref_text_tokens = head.module.tokenizer(ref_texts)
        with torch.no_grad():
            ref_text_token_chunks = torch.split(
                ref_text_tokens, args.experiment.evaluation.batch_size
            )
            for ref_text_token_chunk in tqdm.tqdm(ref_text_token_chunks, desc="Encoding reference texts..."):
                embed = head.module.text_encoder.encode_text(ref_text_token_chunk.to(device))
                ref_text_embed.append(embed)
        self.ref_text_embed = torch.cat(ref_text_embed)
        self.ref_image_batch_size = args.experiment.task.pretraining_batch_size

        self.last_step_backbone_weight = None
        self.last_step_head_weight = None

    @torch.no_grad()
    def forward_ref_model(self, backbone, head, images, texts, **kwargs):
        features = backbone(images)
        head_out = head(texts, features)

        return {"features": features, **head_out}

    def distillation(self, ref_logits, logits, temp=0.02):
        ref_prob = F.softmax(ref_logits / temp, dim=1)
        loss = F.cross_entropy(logits / temp, ref_prob, reduction="mean") * (temp ** 2)
        return loss

    def l2_loss(selfm, model_params, ref_model_weights):
        loss = 0.0
        for param, ref_param in zip(model_params, ref_model_weights.values()):
            loss += F.mse_loss(param, ref_param.to(param.device), reduction="sum")
        return loss

    def ema_merge(self, weights_0, weights_1, step):
        weights_0 = {k: v.cpu() for k, v in weights_0.items()}
        weights_1 = {k: v.cpu() for k, v in weights_1.items()}

        assert set(weights_0.keys()) == set(weights_1.keys()), "Model keys mismatch"

        merged_weights = {
            k: (weights_0[k] + step * weights_1[k]) / (1.0 + step) for k in weights_0
        }
        return merged_weights

    def observe(self, images, targets, **kwargs):
        """Continual Learner Single Training Step
        Args:
            images: [torch.Tensor: BS x C x W x H]
            targets: [torch.Tensor: BS (x 1)]
            output_subset:  [List/torch.Tensor/np.array] - denotes output logit subset to use for training in open-vocabulary continual learning.

        """
        self.opt.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = self.forward(images=images, **kwargs)
            logit_scale = getattr(self.head.module.text_encoder, "logit_scale", 1.0)
            temp = 1.0 / logit_scale.exp()

            if self.ref_image_batch_size == 0:
                base_loss = self.loss(targets=targets, temperature=temp, **outputs, **kwargs)
                loss = base_loss
            else:
                base_loss = self.loss(
                    targets=targets,
                    temperature=temp,
                    features=outputs["features"][:-self.ref_image_batch_size],
                    text_features=outputs["text_features"][:-self.ref_image_batch_size],
                    logits=outputs["logits"][:-self.ref_image_batch_size,
                           :-self.ref_image_batch_size],
                    **kwargs
                )

                ref_images = images[-self.ref_image_batch_size:]
                if self.aux_device != self.device:
                    ref_images = ref_images.to(self.aux_device)
                with torch.no_grad():
                    zs_image_embed = self.zs_backbone(ref_images)
                if self.aux_device != self.device:
                    zs_image_embed = zs_image_embed.to(self.device)
                zs_text_embed = self.ref_text_embed
                zs_logits = (F.normalize(zs_image_embed, dim=-1)
                             @ F.normalize(zs_text_embed, dim=-1).T)

                curr_image_embed = outputs["features"][-self.ref_image_batch_size:]
                curr_logits = (F.normalize(curr_image_embed, dim=-1)
                               @ F.normalize(zs_text_embed, dim=-1).T)

                dist_loss = self.distillation(zs_logits, curr_logits, temp * self.distill_temp)
                dist_loss += self.distillation(zs_logits.T, curr_logits.T, temp * self.distill_temp)

                loss = base_loss + dist_loss

            if self.l2 > 0:
                l2_loss = self.l2_loss(self.backbone.parameters(), self.last_step_backbone_weight)
                l2_loss += self.l2_loss(self.head.parameters(), self.last_step_head_weight)

                loss += self.l2 * l2_loss

        self.gradient_update(loss)

        if (self.iteration + 1) % self.we_freq == 0:
            self.we_step += 1
            self.we_backbone_weights = self.ema_merge(
                self.backbone.state_dict(), self.we_backbone_weights, self.we_step
            )
            self.we_head_weights = self.ema_merge(
                self.head.state_dict(), self.we_head_weights, self.we_step
            )

        self.iteration += 1

        return loss.item()

    def begin_task(
            self,
            optimizer: str = None,
            scheduler: str = None,
            scheduler_steps: int = None,
            milestone_steps: List[int] = None
    ) -> None:
        super().begin_task(optimizer, scheduler, scheduler_steps, milestone_steps)
        self.iteration = 0
        self.we_step = 0
        self.we_backbone_weights = copy.deepcopy(
            {k: v.cpu() for k, v in self.backbone.state_dict().items()}
        )
        self.we_head_weights = copy.deepcopy(
            {k: v.cpu() for k, v in self.head.state_dict().items()}
        )

        if self.l2 > 0:
            self.last_step_backbone_weight = copy.deepcopy(self.we_backbone_weights)
            self.last_step_head_weight = copy.deepcopy(self.we_head_weights)

    @torch.no_grad()
    def end_task(self, experiment, **kwargs):
        if 1 <= self.we_freq <= self.iteration:
            self.backbone.load_state_dict(
                {k: v.cuda() for k, v in self.we_backbone_weights.items()}
            )

            self.head.load_state_dict(
                {k: v.cuda() for k, v in self.we_head_weights.items()}
            )

    @property
    def checkpoint(self):
        return {"self": self.state_dict()}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
