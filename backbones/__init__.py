import abc
import copy
from typing import List

import clip
import numpy as np
import omegaconf
import open_clip
import timm
import torch
import tqdm
import os

########################## Base Variables
# Default timm backbones.
timm_models = [
    "resnet18",
    "resnet50",
    "efficientnet_b2",
    "vit_base_patch32_224",
    "vit_base_patch16_224",
    "vit_large_patch14_224",
]
# CLIP using the Default CLIP backbones.
clip_models = [
    "clip_rn50",
    "clip_rn101",
    "clip_vit_b32",
    "clip_vit_b16",
    "clip_vit_l14",
]
# CLIP using the OpenCLIP backbones.
openclip_models = [
    "openclip_vit_s16",
    "openclip_vit_b32",
    "openclip_vit_b32_datacomp",
    "openclip_vit_b16",
    "openclip_vit_b16_datacomp",
    "openclip_vit_l14",
    "openclip_vit_l14_datacomp",
    "openclip_vit_h14",
    "openclip_vit_g14",
    "openclip_vit_G14",
    "metaclip_vit_b32",
    "metaclip_vit_b16",
    "metaclip_vit_l14",
    "metaclip_vit_h14",
    "dfn_vit_b16",
    "dfn_vit_l14",
    "dfn_vit_h14",
]
# Dictionary comprising model summaries:
# [
#   feature dimensionality,
#   name of optional classification head,
#   list of commonly utilized architecture separators,
#   if available: list of similar separators for a default associated head model - otherwise empty list,
#   if available: patch-generator (for Vision-Transformer models)
model_dict = {
    "highres_resnet18": [512, "fc", ["layer1", "layer2", "layer3", "layer4"], [], None],
    "resnet18": [512, "fc", ["layer1", "layer2", "layer3", "layer4"], [], None],
    "resnet50": [2048, "fc", ["layer1", "layer2", "layer3", "layer4"], [], None],
    "efficientnet_b2": [
        1408,
        "classifier",
        [f"blocks.{i}" for i in range(7)],
        [],
        None,
    ],
    "vit_base_patch16_224": [
        768,
        "head",
        [f"blocks.{i}" for i in range(12)],
        [],
        "patch_embed",
    ],
    "vit_base_patch32_224": [
        768,
        "head",
        [f"blocks.{i}" for i in range(12)],
        [],
        "patch_embed",
    ],
    "vit_large_patch16_224": [
        1024,
        "head",
        [f"blocks.{i}" for i in range(24)],
        [],
        "patch_embed",
    ],
    "clip_rn50": [
        1024,
        None,
        ["layer1", "layer2", "layer3", "layer4"],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        None,
    ],
    "clip_rn101": [
        512,
        None,
        ["layer1", "layer2", "layer3", "layer4"],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        None,
    ],
    "clip_vit_b32": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "clip_vit_b16": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "clip_vit_l14": [
        768,
        None,
        [f"transformer.resblocks.{i}" for i in range(24)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "openclip_vit_s16": [
        384,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "openclip_vit_b32": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "openclip_vit_b32_datacomp": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "openclip_vit_b16": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "openclip_vit_b16_datacomp": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "openclip_vit_l14": [
        768,
        None,
        [f"transformer.resblocks.{i}" for i in range(24)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "openclip_vit_l14_datacomp": [
        768,
        None,
        [f"transformer.resblocks.{i}" for i in range(24)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "openclip_vit_h14": [
        1024,
        None,
        [f"transformer.resblocks.{i}" for i in range(32)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(24)],
        "conv1",
    ],
    "openclip_vit_g14": [
        1024,
        None,
        [f"transformer.resblocks.{i}" for i in range(40)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(24)],
        "conv1",
    ],
    "openclip_vit_G14": [
        1280,
        None,
        [f"transformer.resblocks.{i}" for i in range(48)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(32)],
        "conv1",
    ],
    "metaclip_vit_b32": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "metaclip_vit_b16": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "metaclip_vit_l14": [
        768,
        None,
        [f"transformer.resblocks.{i}" for i in range(24)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "metaclip_vit_h14": [
        1024,
        None,
        [f"transformer.resblocks.{i}" for i in range(32)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(24)],
        "conv1",
    ],
    "dfn_vit_b16": [
        512,
        None,
        [f"transformer.resblocks.{i}" for i in range(12)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "dfn_vit_l14": [
        768,
        None,
        [f"transformer.resblocks.{i}" for i in range(24)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(12)],
        "conv1",
    ],
    "dfn_vit_h14": [
        1024,
        None,
        [f"transformer.resblocks.{i}" for i in range(32)],
        [f"text_encoder.transformer.resblocks.{i}" for i in range(24)],
        "conv1",
    ],
}

# Dictionary comprising output feature sizes for all models:
feature_dim_dict = {key: item[0] for key, item in model_dict.items()}
# Dictionary comprising names of feature-to-output-maps (e.g. to logits) if available.
default_heads = {key: item[1] for key, item in model_dict.items()}
# Dictionary comprising lists of commonly utilized architecture separators.
backbone_separators = {key: item[2] for key, item in model_dict.items()}
# Dictionary comprising lists of commonly utilized architecture separators for optional default head architectures (e.g. text encoders in CLIP-style models).
head_separators = {key: item[3] for key, item in model_dict.items()}
# Dictionary comprising patch-generator module names if available.
patch_modules = {key: item[4] for key, item in model_dict.items()}

### Complete list of all backbones and allowed head modes.
BACKBONES = [
    # Adapted ResNet used in the default mammoth setup with reduced downsampling for lowres datasets such as CIFAR
    "highres_resnet18"
]
BACKBONES.extend(timm_models)
BACKBONES.extend(clip_models)
BACKBONES.extend(openclip_models)
# Possible types of model-heads that can be attached to feature outputs.
#   -> semantic_{model_name} means that we embed all classnames and freeze the resulting weight matrix.
HEADS = ["default", "linear", "mlp-2", "mlp-3"] + [
    "semantic_" + x for x in clip_models + openclip_models
]

### Dataset Statistics.
# ImageNet Stats.
IN1K_MEAN = [0.485, 0.456, 0.406]
IN1K_STD = [0.229, 0.224, 0.225]
# CLIP Stats.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

clip_conv = {
    "clip_rn50": "RN50",
    "clip_rn101": "RN101",
    "clip_vit_b32": "ViT-B/32",
    "clip_vit_b16": "ViT-B/16",
    "clip_vit_l14": "ViT-L/14",
}

openclip_conv = {
    "openclip_vit_s16": ["ViT-S-16", "ViT-S-16_datacomp1b_lr0.001_b1_0.9_b2_0.95_wd0.2_warm4000_bs90k_constcooldown_s12.8B_inet66.2.pt"],  # specifically for the S-16 model we have to load the ckpt directly, please download this model using `wget https://huggingface.co/mehdidc/ViT-S-16_datacomp1b_lr0.001_b1_0.9_b2_0.95_wd0.2_warm4000_bs90k_constcooldown_s12.8B_inet66.2/resolve/main/ViT-S-16_datacomp1b_lr0.001_b1_0.9_b2_0.95_wd0.2_warm4000_bs90k_constcooldown_s12.8B_inet66.2.pt` and place it in the correct cache-directory so that the script can access it.
    "openclip_vit_b32": ["ViT-B-32", "laion2b_s34b_b79k"],
    "openclip_vit_b32_datacomp": ["ViT-B-32", "datacomp_xl_s13b_b90k"],
    "openclip_vit_b16": ["ViT-B-16", "laion2b_s34b_b88k"],
    "openclip_vit_b16_datacomp": ["ViT-B-16", "datacomp_xl_s13b_b90k"],
    "openclip_vit_l14": ["ViT-L-14", "laion2b_s32b_b82k"],
    "openclip_vit_l14_datacomp": ["ViT-L-14", "datacomp_xl_s13b_b90k"],
    "openclip_vit_h14": ["ViT-H-14", "laion2b_s32b_b79k"],
    "openclip_vit_g14": ["ViT-g-14", "laion2b_s34b_b88k"],
    "openclip_vit_G14": ["ViT-bigG-14", "laion2b_s39b_b160k"],
    "metaclip_vit_b32": ["ViT-B-32-quickgelu", "metaclip_fullcc"],
    "metaclip_vit_b16": ["ViT-B-16-quickgelu", "metaclip_fullcc"],
    "metaclip_vit_l14": ["ViT-L-14-quickgelu", "metaclip_fullcc"],
    "metaclip_vit_h14": ["ViT-H-14-quickgelu", "metaclip_fullcc"],
    "dfn_vit_b16": ["ViT-B-16", "dfn2b"],
    "dfn_vit_l14": ["ViT-L-14-quickgelu", "dfn2b"],
    "dfn_vit_h14": ["ViT-H-14-quickgelu", "dfn5b"],
}


########################## Key Functions
class Dummy(abc.ABC):
    def __init__(self):
        pass


def convert_model_to_fp32(model: torch.nn.Module):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def get_backbone(
    device: torch.device,
    backbone_name: str,
    pretrained: bool,
    half_precision: bool,
    cache_dir: str = "./cache_dir",
):
    optional_text_encoder = None

    ### Custom High-resolution ResNet18 for standard CL benchmarks.
    if backbone_name == "highres_resnet18":
        if pretrained:
            raise NotImplementedError("HighRes ResNet18 has no pretraining available!")
        # We use a default logit count of 2, will be overwritten later in the continual learner.
        visual_backbone = backbones.highres_resnet18.resnet18(nclasses=2)

    ### (ImageNet-pretrained) Default TIMM architectures.
    if backbone_name in timm_models:
        visual_backbone = timm.create_model(backbone_name, pretrained=pretrained)
        visual_backbone.mean, visual_backbone.std = IN1K_MEAN, IN1K_STD

    ### Default OAI CLIP architectures.
    if backbone_name in clip_models:
        backbone, _ = clip.load(
            clip_conv[backbone_name], device, jit=False, download_root=cache_dir
        )
        # torch.cuda.amp.autocast can not rescale fp16 gradients,
        # hence even for half/mixed-precision training we rescale model weights.
        convert_model_to_fp32(backbone)

    ### (LAION-pretrained) Default OpenCLIP architectures.
    if backbone_name in openclip_models:
        if 'vit_s16' in backbone_name:
            # loading for ViT-S model directly from checkpoint
            # first load randomly init ViT-S-16
            backbone, _, _ = open_clip.create_model_and_transforms(openclip_conv[backbone_name][0])
            # second load the torch checkpoint
            ckpt = torch.load(os.path.join(cache_dir, openclip_conv[backbone_name][1]))
            model_ckpt = {k.replace('module.', ''):v for k,v in dict(ckpt['state_dict']).items()}
            # finally load the ckpt into the model
            backbone.load_state_dict(model_ckpt)
        else:
            # standard loading
            backbone, _, _ = open_clip.create_model_and_transforms(
                openclip_conv[backbone_name][0],
                pretrained=openclip_conv[backbone_name][1],
                cache_dir=cache_dir,
            )
        backbone.mean, backbone.std = CLIP_MEAN, CLIP_STD
        _ = backbone.to(device)

    if backbone_name in openclip_models:
        visual_backbone = copy.deepcopy(backbone.visual)
        visual_backbone.mean, visual_backbone.std = CLIP_MEAN, CLIP_STD
        del backbone.visual
        optional_text_encoder = backbone

    if backbone_name in clip_models:
        visual_backbone = copy.deepcopy(backbone.visual)
        visual_backbone.mean, visual_backbone.std = CLIP_MEAN, CLIP_STD
        # We need to retain part of the visual component for the text encoder to avoid weird dtype errors.
        optional_text_encoder = backbone
        del optional_text_encoder.visual.transformer
        for param in optional_text_encoder.visual.parameters():
            param.requires_grad = False

    return visual_backbone, optional_text_encoder


def get_head(
    device: torch.device,
    head_name: str,
    backbone_name: str,
    backbone: torch.nn.Module,
    pretrained: bool,
    classnames: List[str],
    half_precision: bool,
    cache_dir: str = "./cache_dir",
):
    ### Load model head
    # Note: This can both be your standard linear/MLP projection on top of vision features for classification,
    # or reference the text-encoder in CLIP-style setups.
    if head_name == "default":
        assert_str = "highres_resnet18 has no default (pretrained) classification head."
        assert backbone_name != "highres_resnet18", assert_str

        if backbone_name in timm_models:
            head = LinearHead(head=getattr(backbone, default_heads[backbone_name]))
        if backbone_name in clip_models or backbone_name in openclip_models:
            head = ClipTextHead(
                device, backbone_name, backbone, "openclip" in backbone_name
            )

    if head_name == "linear":
        head = LinearHead(feature_dim_dict[backbone_name], len(classnames))

    if "mlp" in head_name:
        depth = int(head_name.split("-")[-1])
        head = MlpHead(feature_dim_dict[backbone_name], len(classnames), depth)

    if "semantic" in head_name:
        text_encoder_name = head_name.split("semantic_")[-1]
        if text_encoder_name == backbone_name:
            text_encoder = backbone
        else:
            text_encoder = get_backbone(
                device, text_encoder_name, pretrained, half_precision, cache_dir
            )
        head = SemanticHead(
            device,
            text_encoder_name,
            text_encoder,
            classnames,
            "openclip" in backbone_name,
        )

    return head


def get_backbone_and_head(
    device: torch.device, args: omegaconf.DictConfig, classnames: List[str] = None
):
    backbone_name = args.experiment.backbone.name
    assert backbone_name in BACKBONES, f"No backbone {backbone_name} available."

    head_name = args.experiment.backbone.head
    assert head_name in HEADS, f"No head-type {head_name} available."

    half_precision = args.experiment.backbone.half_precision
    cache_dir = args.experiment.backbone.cache_dir
    pretrained = args.experiment.backbone.pretrained

    ### Load up Vision-Backbones
    backbone, optional_text_encoder = get_backbone(
        device, backbone_name, pretrained, half_precision, cache_dir
    )

    ### Load corresponding classification / text-embedding head.
    if "semantic" in head_name:
        assert_str = f"Semantic heads ({args.experiment.backbone.head}) not applicable to training == contrastive!"
        assert args.experiment.training != "contrastive", assert_str
    head = get_head(
        device,
        head_name,
        backbone_name,
        optional_text_encoder,
        pretrained,
        classnames,
        half_precision,
        cache_dir,
    )

    ### In case resizing and different renormalization of used datasets is required because of pretrained = True.
    dataloader_updates = {}
    if pretrained:
        dataloader_updates = {"mean": backbone.mean, "std": backbone.std}

    ## perform freezing operations as required
    if args.experiment.backbone.freeze_head:
        # freeze the head
        for _, w in head.named_parameters():
            w.requires_grad = False

    if args.experiment.backbone.freeze_features:
        # freeze the visual backbone
        for _, w in backbone.named_parameters():
            w.requires_grad = False

    return backbone, head, dataloader_updates


########################## Available Head Types


class LinearHead(torch.nn.Module):
    def __init__(
        self,
        in_features: int = None,
        num_classes: int = None,
        head: torch.nn.Module = None,
    ):
        super().__init__()
        if head is not None:
            self.head = head
        else:
            self.head = torch.nn.Linear(in_features, num_classes)

    def forward(self, features, **kwargs):
        return {"logits": self.head(features)}


class MlpHead(torch.nn.Module):
    def __init__(
        self,
        in_features: int = None,
        num_classes: int = None,
        depth: int = 2,
        head: torch.nn.Module = None,
    ):
        super().__init__()
        if head is not None:
            self.head = head
        else:
            head_list = [torch.nn.Linear(in_features, in_features)]
            for i in range(depth - 2):
                head_list.append(torch.nn.Linear(in_features, in_features))
            head_list.append(torch.nn.Linear(in_features, num_classes))
            self.head = torch.nn.Sequential(*head_list)

    def forward(self, features, **kwargs):
        return {"logits": self.head(features)}


class SemanticHead(torch.nn.Module):
    """
    This provides a simple linear head, with each row corresponding to the language embedding of each classname.
    """

    def __init__(
        self,
        device: torch.device,
        text_encoder_name: str,
        text_encoder: torch.nn.Module,
        classnames: List[str],
        openclip: bool = True,
    ):
        super().__init__()
        self.device = device
        if openclip:
            self.tokenizer = open_clip.get_tokenizer(
                openclip_conv[text_encoder_name][0]
            )
        else:
            self.tokenizer = clip.tokenize

        # Tokenize classnames.
        with torch.cuda.amp.autocast(), torch.no_grad():
            text_tokens = self.tokenizer(classnames).cuda()

        self.head = []
        batch_size = 128
        num_batches = int(np.ceil(len(text_tokens) / batch_size))

        # Compute classname embeddings
        with torch.cuda.amp.autocast(), torch.no_grad():
            for i in tqdm.tqdm(range(num_batches), desc="Encoding text features..."):
                self.head.append(
                    torch.nn.functional.normalize(
                        text_encoder.encode_text(
                            text_tokens[i * batch_size : (i + 1) * batch_size]
                        ),
                        dim=-1,
                    )
                )

        self.head = torch.nn.Parameter(torch.cat(self.head, dim=0)).cuda()

    def forward(self, features: torch.Tensor, **kwargs):
        logits = torch.nn.functional.normalize(features, dim=-1) @ self.head.T
        return {"logits": logits}


class ClipTextHead(torch.nn.Module):
    """
    This provides the standard language side of CLIP, which takes in at least corresponding text lists and returns respective text embeddings.
    If image features are provided, it also jointly computes the corresponding logits / similarities.
    """

    def __init__(
        self,
        device: torch.device,
        text_encoder_name: str,
        text_encoder: torch.nn.Module,
        openclip: bool = True,
    ):
        super().__init__()
        self.device = device
        if openclip:
            self.tokenizer = open_clip.get_tokenizer(
                openclip_conv[text_encoder_name][0]
            )
        else:
            self.tokenizer = clip.tokenize
        self.text_encoder = text_encoder.cuda()

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        grads = []
        for pp in list(self.parameters()):
            if pp.grad is not None:
                grads.append(pp.grad.view(-1))
            else:
                grads.append(torch.zeros_like(pp).view(-1))
        return grads

    def embed_text(self, texts, batch_size):
        embed_coll = []

        # Compute classname embeddings
        text_tokens = self.tokenizer(texts).cuda()

        num_batches = int(np.ceil(len(text_tokens) / batch_size))
        for i in range(num_batches):
            embed_coll.append(
                torch.nn.functional.normalize(
                    self.text_encoder.encode_text(
                        text_tokens[i * batch_size : (i + 1) * batch_size]
                    ),
                    dim=-1,
                )
            )
        return torch.cat(embed_coll, dim=0)

    def forward(self, texts, features: torch.Tensor = None, **kwargs):
        text_tokens = self.tokenizer(texts).cuda()

        text_features = torch.nn.functional.normalize(
            self.text_encoder.encode_text(text_tokens), dim=-1
        )
        logits = None
        if features is not None:
            logits = torch.nn.functional.normalize(features, dim=-1) @ text_features.T
        return {"text_features": text_features, "logits": logits}
