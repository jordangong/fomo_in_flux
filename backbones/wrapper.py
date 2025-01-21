import copy
from typing import Union, List

import torch


class ForwardHook:
    def __init__(
        self,
        hook_dict,
        layer_name: str,
        last_layer_to_extract: str,
        stop_when_last_layer_reached: bool = False,
    ):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.stop_when_last_layer_reached = stop_when_last_layer_reached
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.stop_when_last_layer_reached and self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class ExtractionWrapper(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        num_classes: Union[List[str], int] = None,
        output_hook: str = "fc",
        feature_hook: str = "avgpool",
        other_hooks: List[str] = [],
        replace_output: bool = True,
    ):
        super(ExtractionWrapper, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.output_hook = output_hook
        self.feature_hook = feature_hook
        self.other_hooks = other_hooks
        self.hooks = [self.feature_hook] + [self.output_hook] + self.other_hooks

        self.replace_output = replace_output

        # Only return a subset of the output-tensor (e.g. when only a subset of classes/concepts have been seen).
        # If so, this value can be set from the outside to a class/concept index list.
        self.output_subset = None

        self._set_hooks()

        if hasattr(self.backbone, "class_names"):
            self.class_names = self.backbone.class_names
        else:
            self.class_names = None

        self.warmup = False

    @property
    def active_concepts(self):
        if self.output_subset is None:
            return self.class_names
        else:
            return [self.class_names[i] for i in self.output_subset]

    @property
    def classifier(self):
        return self.backbone.__dict__["_modules"][self.output_hook]

    def _replace_classifier(self, new_classifier_head):
        # TODO: This needs checking!
        self.replace_output = False
        self.backbone.__dict__["_modules"][self.output_hook] = new_classifier_head
        self._set_hooks()

    def _set_hooks(self):
        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()

        self.outputs = {}
        for hook in self.hooks:
            hook_name, base_hook, subhook, prev_mod_dict, hook_layer = self.select_hook(
                hook
            )

            forward_hook = ForwardHook(self.outputs, hook_name, self.hooks[-1])

            if (
                isinstance(self.num_classes, int)
                and self.replace_output
                and base_hook == self.output_hook
            ):
                prev_mod_dict[subhook] = torch.nn.Linear(
                    hook_layer.in_features, self.num_classes
                )
                hook_layer = prev_mod_dict[subhook]

            if isinstance(hook_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    hook_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    hook_layer.register_forward_hook(forward_hook)
                )

    def select_hook(self, hook):
        if hook == self.output_hook:
            hook_name = "out"
        elif hook == self.feature_hook:
            hook_name = "features"
        else:
            hook_name = hook

        base_hook = hook
        if "." in hook:
            hook = hook.split(".")
        else:
            hook = [hook]

        hook_layer = self.backbone
        for subhook in hook:
            prev_mod_dict = hook_layer.__dict__["_modules"]
            hook_layer = hook_layer.__dict__["_modules"][subhook]

        return hook_name, base_hook, subhook, prev_mod_dict, hook_layer

    def warmup_on(self):
        pass
        # if not self.warmup:
        #     _, _, _, _, hook_layer = self.select_hook(self.feature_hook)
        #     self.warmup = True
        #     self.warmup_handle = None
        #     from IPython import embed; embed()

    def warmup_off(self):
        pass
        # if self.warmup:
        #     self.warmup = False
        #     pass

    def forward(
        self, x: torch.Tensor, output_subset: List[int] = None, returnt="out"
    ) -> torch.Tensor:
        if output_subset is None:
            output_subset = self.output_subset
        subsetting_handled_by_backbone = (
            "output_subset" in self.backbone.forward.__code__.co_varnames
        )
        if subsetting_handled_by_backbone:
            _ = self.backbone(x, output_subset)
            if isinstance(returnt, list):
                return [self.outputs[return_key] for return_key in returnt]
            else:
                return self.outputs[returnt]
        else:
            _ = self.backbone(x)
            if isinstance(returnt, list):
                out_list = []
                for return_key in returnt:
                    if output_subset is not None and return_key == "out":
                        out_list.append(self.outputs[return_key][..., output_subset])
                    else:
                        out_list.append(self.outputs[return_key])
                return out_list
            else:
                if output_subset is not None:
                    return self.outputs[returnt][..., output_subset]
                else:
                    return self.outputs[returnt]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.backbone(x)
        output = self.outputs["features"]
        if self.hook_subset_idcs["features"] is not None:
            return output[..., self.hook_subset_idcs["features"]]
        else:
            return output

    def get_params_list(self, exclude_classifier=False) -> List[torch.Tensor]:
        params = []
        for n, p in self.named_parameters():
            if exclude_classifier and (
                f"backbone.{self.output_hook}" in n or "_fc" in n
            ):
                pass
            else:
                params.append(p)
        return params

    def get_params(self) -> torch.Tensor:
        params = []
        for p in list(self.parameters()):
            params.append(p.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for p in list(self.parameters()):
            cand_params = new_params[
                progress : progress + torch.tensor(p.size()).prod()
            ].view(p.size())
            progress += torch.tensor(p.size()).prod()
            p.data = cand_params

    def get_grads(self) -> torch.Tensor:
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads
