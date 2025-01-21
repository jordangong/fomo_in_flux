# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import continual_lib

import torch


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(self, args, backbone, head, loss, device, **kwargs):
        super(Model, self).__init__(args, backbone, head, loss, device)

    def observe(self, images, targets, **kwargs):
        """Continual Learner Single Training Step
        Args:
            images: [torch.Tensor: BS x C x W x H]
            targets: [torch.Tensor: BS (x 1)]
            output_subset:  [List/torch.Tensor/np.array] - denotes output logit subset to use for training in open-vocabulary continual learning.

        """
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
