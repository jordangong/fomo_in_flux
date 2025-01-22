import numpy as np
import torch

import continual_lib
from continual_lib.utils import buffer


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(self, args, backbone, head, loss, device, experiment, alpha, **kwargs):
        super(Model, self).__init__(args, backbone, head, loss, device)

        assert_str = "Used buffer.size = 0 for buffer-based continual.method=[er]!"
        assert args.experiment.buffer.size > 0, assert_str

        self.buffer = buffer.Buffer(
            args.experiment.buffer.size,
            args.experiment.buffer.batch_size,
            device,
            training_mode=self.training_mode,
            transform=(
                None
                if not args.experiment.buffer.with_transform
                else experiment.train_transform
            ),
        )
        self.alpha = alpha

    def observe(self, images, targets, **kwargs):
        self.opt.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = self.forward(images=images, **kwargs)
            loss = self.loss(targets=targets, **outputs, **kwargs)

            if not self.buffer.is_empty():
                buf_data = self.buffer.get_data(**kwargs)
                buf_outputs = self.forward(**buf_data, experiment=kwargs["experiment"])

                buf_loss = self.loss(
                    targets=buf_data["targets"], **buf_outputs, **kwargs
                )

                loss += self.alpha * buf_loss

        self.gradient_update(loss)

        buffer_data = {
            "images": (
                images
                if "non_aug_inputs" not in kwargs or kwargs["non_aug_inputs"] is None
                else kwargs["non_aug_inputs"]
            ),
            "targets": targets,
            **outputs,
            **{key: item for key, item in kwargs.items() if key not in ["experiment"]},
        }
        self.buffer.add_data(**buffer_data)

        return loss.item()

    @property
    def checkpoint(self):
        return {"self": self.state_dict(), "buffer": self.buffer.checkpoint}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
        self.buffer.load_checkpoint(state_dict["buffer"])
