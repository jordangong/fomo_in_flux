import omegaconf
import torch

import backbones


#### Utilities
class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, hook_input: bool = False):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.hook_input = hook_input

    def __call__(self, module, input, output):
        if not self.hook_input:
            self.hook_dict[self.layer_name] = output
        else:
            self.hook_dict[self.layer_name] = input[0]
        return None


def hook_default_features(
    args: omegaconf.DictConfig, hook_object: torch.nn.Module, hook_dict: dict
):
    possible_head = backbones.default_heads[args.experiment.backbone.name]
    if not possible_head:
        forward_hook = ForwardHook(hook_dict, "features")
        return hook_object.register_forward_hook(forward_hook)
    else:
        forward_hook = ForwardHook(hook_dict, "features", hook_input=True)
        return getattr(hook_object, possible_head).register_forward_hook(forward_hook)
