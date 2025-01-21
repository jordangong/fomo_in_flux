# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
from typing import Any

import omegaconf
import torch

from continual_lib.utils.base_continual_learner import BaseContinualLearner

ALLOWED_TRAINING = BaseContinualLearner.SUPPORTED_TRAINING_MODES


def get_all_methods():
    return sorted(
        [
            method.split(".")[0]
            for method in os.listdir("continual_lib")
            if not method.find("__") > -1 and "py" in method
        ]
    )


METHODS = get_all_methods()


def get_class(args):
    assert (
        args.continual.method in METHODS
    ), f"No method {args.continual.method} available."
    mod = importlib.import_module(f"continual_lib.{args.continual.method}")
    return getattr(mod, "Model")


def get(
    args: omegaconf.DictConfig,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    loss: Any,
    device: torch.device,
    experiment: Any,
    params: dict,
):
    mod = get_class(args)
    return mod(args, backbone, head, loss, device, experiment=experiment, **params)


def get_loss(args):
    if args.experiment.optimizer.loss == "cross_entropy":
        assert_str = (
            "loss [cross_entropy] currently not available for training=[contrastive]!"
        )
        assert args.experiment.training != "contrastive", assert_str
        return CrossEntropyLoss(
            label_smoothing=args.experiment.optimizer.label_smoothing
        )
    elif args.experiment.optimizer.loss == "clip":
        return CLIPLoss(
            temperature=args.experiment.optimizer.clip_temperature
        )
    elif args.experiment.optimizer.loss == "classclip":
        assert_str = (
            "loss [classclip] currently only available for training=[contrastive]!"
        )
        assert args.experiment.training == "contrastive", assert_str
        return ClassCLIPLoss(
            temperature=args.experiment.optimizer.clip_temperature,
            label_smoothing=args.experiment.optimizer.label_smoothing,
        )
    else:
        raise NotImplementedError(
            f"Loss {args.experiment.optimizer.loss} not available."
        )


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, label_smoothing):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def forward(self, logits, targets, **kwargs):
        return self.loss(logits, targets)


class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature: float):
        """Compute CLIP loss

        For both contrastive and classification-based training.

        Args:
            temperature (float): similarity temperature.
            training_mode (str): contrastive vs classification.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        **kwargs,
    ):
        """_summary_

        Args:
            logits (torch.Tensor): _description_
            targets (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            loss: Final Cliploss
        """
        clip_targets = torch.arange(logits.size(1), device=logits.device)

        temp_to_use = (
            temperature if self.temperature == "learnable" else self.temperature
        )

        loss = torch.nn.functional.cross_entropy(
            logits / temp_to_use, clip_targets
        )
        loss += torch.nn.functional.cross_entropy(
            logits.T / temp_to_use, clip_targets
        )

        return loss / 2


class ClassCLIPLoss(torch.nn.Module):
    def __init__(
        self, temperature: float, label_smoothing: float, class_mode: str = "xe"
    ):
        """Compute Classification + CLIP loss

        For contrastive atraining.

        Args:
            temperature (float): similarity temperature.
        """
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.class_mode = class_mode
        if self.class_mode == "xe":
            self.miner, self.loss = None, CrossEntropyLoss(self.label_smoothing)
        else:
            raise NotImplementedError()
            # from pytorch_metric_learning import miners, losses
            # self.miner, self.loss = miners.MultiSimilarityMiner(), losses.MultiSimilarityLoss()

        self.clip_loss = CLIPLoss(self.temperature, "contrastive")

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor = None,
        text_features: torch.Tensor = None,
        targets: torch.Tensor = None,
        **kwargs,
    ):
        """_summary_

        Args:
            logits (torch.Tensor): _description_
            features (torch.Tensor, optional): _description_. Defaults to None.
            text_features (torch.Tensor, optional): _description_. Defaults to None.
            targets (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            loss: Final Cliploss
        """
        # Perform class-level contrastive training.
        # Supervised loss minimizing similarity of
        # - same-class image embeddings.
        # - same-class text embeddings.
        # - same-class image-text pairs.
        features = torch.nn.functional.normalize(features, dim=-1)
        image_text_sims = logits
        image_sims = features @ features.T
        text_sims = text_features @ text_features.T
        sims_list = [image_text_sims, image_text_sims.T, image_sims, text_sims]
        mul_list = [0.5, 0.5, 1, 1]

        weighted_positives = (targets.view(-1, 1) == targets.view(1, -1)).type(
            torch.float
        )
        weighted_positives /= weighted_positives.sum(dim=-1)
        supervised_loss = (
            torch.sum(
                torch.tensor(
                    [
                        self.loss(sims, weighted_positives) * mul
                        for mul, sims in zip(mul_list, sims_list)
                    ]
                )
            )
            / 3
        )

        # Default CLIP loss - perform sample-level contrastive training.
        clip_loss = self.clip_loss(image_text_sims)

        return (supervised_loss + clip_loss) / 2

class CLIP4ClassLoss(torch.nn.Module):
    def __init__(self, temperature: float):
        """Compute CLIP4Class loss

        For classification-based training.

        Args:
            temperature (float): similarity temperature.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        """_summary_

        Args:
            logits (torch.Tensor): _description_
            targets (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            loss: Final Cliploss
        """
        temp_to_use = temperature if self.temperature == 'learnable' else self.temperature
        # Wrt to classification targets.
        # Logits have shape BS x NUM_CLASSES.
        # We compute a cross-entropy loss alongside the target axis, and a weighted XE-loss alongside the batch axis.
        loss = torch.nn.functional.cross_entropy(logits / temp_to_use, targets)
        subidcs = torch.where(torch.logical_not(torch.isinf(logits[0])))[0]
        adjusted_targets = (
            (targets.view(-1, 1) == clip_targets.view(1, -1)).type(torch.float).T
        )
        adjusted_targets /= (
            torch.sum(adjusted_targets, dim=1).view(-1, 1).clip(1, None)
        )
        loss += torch.nn.functional.cross_entropy(
            logits[:, subidcs].T / temp_to_use, adjusted_targets[subidcs]
        )
        return loss / 2
    