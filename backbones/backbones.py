import backbone.foundation_models
import backbone.wrapper
import timm
import torch

import utils.conf


def resnet18(
    class_names: list[str] = None,
    num_classes: int = None,
    other_hooks: list = [],
    pretrained: bool = False,
    replace_output: bool = True,
) -> torch.nn.Module:
    model_backbone = backbone.wrapper.ExtractionWrapper(
        num_classes=num_classes,
        backbone=timm.create_model("resnet18", pretrained=pretrained),
        output_hook="fc",
        feature_hook="global_pool",
        other_hooks=other_hooks,
        replace_output=replace_output,
    ).to(utils.conf.get_device())
    model_backbone.pretrained = pretrained
    model_backbone.pretraining = "IN1k" if pretrained else "None"
    return model_backbone


def resnet50(
    class_names: list[str] = None,
    num_classes: int = None,
    other_hooks: list = [],
    pretrained: bool = False,
    replace_output: bool = True,
) -> torch.nn.Module:
    model_backbone = backbone.wrapper.ExtractionWrapper(
        num_classes=num_classes,
        backbone=timm.create_model("resnet50", pretrained=pretrained),
        output_hook="fc",
        feature_hook="global_pool",
        other_hooks=other_hooks,
        replace_output=replace_output,
    ).to(utils.conf.get_device())
    model_backbone.pretrained = pretrained
    model_backbone.pretraining = "IN1k" if pretrained else "None"
    return model_backbone


def efficientnet_b2(
    class_names: list[str] = None,
    num_classes: int = None,
    other_hooks: list = [],
    pretrained: bool = False,
    replace_output: bool = True,
) -> torch.nn.Module:
    model_backbone = backbone.wrapper.ExtractionWrapper(
        num_classes=num_classes,
        backbone=timm.create_model("efficientnet_b2", pretrained=pretrained),
        output_hook="classifier",
        feature_hook="global_pool",
        other_hooks=other_hooks,
        replace_output=replace_output,
    ).to(utils.conf.get_device())
    model_backbone.pretrained = pretrained
    model_backbone.pretraining = "IN1k" if pretrained else "None"
    return model_backbone


def clip_vit_b32(
    class_names: list[str] = None,
    num_classes: int = None,
    other_hooks: list = [],
    pretrained: bool = False,
    replace_output: bool = False,
) -> torch.nn.Module:
    device = utils.conf.get_device()
    model_backbone = backbone.wrapper.ExtractionWrapper(
        num_classes=num_classes,
        backbone=backbone.foundation_models.CLIP(
            class_names, backbone="ViT-B/32", random_init=pretrained, device=device
        ),
        output_hook="text_features",
        feature_hook="backbone.visual",
        other_hooks=other_hooks,
        replace_output=replace_output,
    ).to(device)
    model_backbone.pretrained = pretrained
    model_backbone.pretraining = "CLIP" if pretrained else "None"
    return model_backbone


def clip_resnet50(
    class_names: list[str] = None,
    num_classes: int = None,
    other_hooks: list = [],
    pretrained: bool = False,
    replace_output: bool = False,
) -> torch.nn.Module:
    device = utils.conf.get_device()
    model_backbone = backbone.wrapper.ExtractionWrapper(
        num_classes=num_classes,
        backbone=backbone.foundation_models.CLIP(
            class_names, backbone="RN50", random_init=pretrained, device=device
        ),
        output_hook="text_features",
        feature_hook="backbone.visual",
        other_hooks=other_hooks,
        replace_output=replace_output,
    ).to(device)
    model_backbone.pretrained = pretrained
    model_backbone.pretraining = "CLIP" if pretrained else "None"
    return model_backbone
