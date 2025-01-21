import numpy as np
import termcolor
import torch

import continual_lib
from continual_lib.utils import buffer


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(
        self, args, backbone, head, loss, device, experiment, alpha, beta, **kwargs
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)

        assert_str = "Used buffer.size = 0 for buffer-based continual.method=[derpp]!"
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
        self.beta = beta
        if self.training_mode == "classification_task":
            termcolor.cprint(
                "Warning: continual.method=[derpp] and training=[classification_task] technically work, but will have limited benefits",
                "yellow",
                attrs=[],
            )

    def observe(self, images, targets, **kwargs):
        self.opt.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = self.forward(images=images, **kwargs)
            loss = self.loss(targets=targets, **outputs, **kwargs)

            if not self.buffer.is_empty():
                buf_data = self.buffer.get_data()
                buf_outputs = self.forward(**buf_data, experiment=kwargs["experiment"])

                # Mask out potential infinite entries in the buffer / buffer-output due to masking.
                non_inf_mask = torch.logical_not(
                    torch.isinf(buf_outputs["logits"])
                ) * torch.logical_not(torch.isinf(buf_data["logits"]))
                val_idcs = torch.where(torch.sum(non_inf_mask, dim=-1) > 0)[0]

                # Compute DER loss.
                buf_loss = 0
                for val_idx in val_idcs:
                    buf_loss += torch.nn.functional.mse_loss(
                        buf_outputs["logits"][val_idx, non_inf_mask[val_idx]],
                        buf_data["logits"][val_idx, non_inf_mask[val_idx]],
                    )
                buf_loss /= np.max([1, len(val_idcs)])
                loss += self.alpha * buf_loss

                # Compute DER++ loss (c.f. ER).
                loss += self.beta * self.loss(
                    targets=buf_data["targets"], **buf_outputs, **kwargs
                )

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
