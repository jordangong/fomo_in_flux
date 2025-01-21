import torch

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
        ln_select: str = "all",
        tune_logit_scale=False,
        **kwargs
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)

        self.ln_select = ln_select

        ### Define adapter parameters.
        for layer, weight in self.head.module.named_parameters():
            mod = self.head.module
            if "ln_" in layer:
                mod_layer = ".".join(layer.split(".")[:-1])
                for name in mod_layer.split("."):
                    mod = mod._modules[name]

                if self.ln_select == "all":
                    weight.requires_grad = True
                else:
                    if self.ln_select in mod_layer:
                        weight.requires_grad = True
                    else:
                        weight.requires_grad = False
            else:
                weight.requires_grad = False

        for layer, weight in self.backbone.module.named_parameters():
            mod = self.backbone.module
            if "ln_" in layer:
                mod_layer = ".".join(layer.split(".")[:-1])
                for name in mod_layer.split("."):
                    mod = mod._modules[name]

                if self.ln_select == "all":
                    weight.requires_grad = True
                else:
                    if self.ln_select in mod_layer:
                        weight.requires_grad = True
                    else:
                        weight.requires_grad = False
            else:
                weight.requires_grad = False

        if tune_logit_scale:
            self.head.module.text_encoder.logit_scale.requires_grad = True

    def observe(self, images, targets, **kwargs):
        self.opt.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = self.forward(images=images, **kwargs)
            logit_scale = getattr(self.head.module.text_encoder, "logit_scale", 1.0)
            temp = 1.0 / logit_scale.exp()
            loss = self.loss(targets=targets, temperature=temp, **outputs, **kwargs)

        self.gradient_update(loss)

        return loss.item()

    @property
    def checkpoint(self):
        return {"self": self.state_dict()}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
