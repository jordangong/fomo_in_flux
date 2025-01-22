import numpy as np
import torch

import continual_lib
from continual_lib.utils import buffer


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(self, args, backbone, head, loss, device, experiment):
        super(Model, self).__init__(args, backbone, head, loss, device)
        self.buffer = buffer.Buffer(
            args.experiment.buffer.size,
            args.experiment.buffer.batch_size,
            device,
            None,
            training_mode=self.training_mode,
        )

    def observe(self, inputs, targets, non_aug_inputs=None, **kwargs):
        self.opt.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = self.backbone(inputs)
            loss = self.loss(outputs, targets)

            if not self.buffer.is_empty():
                buf_data = self.buffer.get_data(**kwargs)

                if self.training_mode == "open":
                    # For open-class continual learning, i.e. we only use classes we have seen before,
                    # we get output logits used across buffer outputs.
                    buf_outputs = self.backbone(
                        buf_data["examples"],
                        output_subset=buf_data["flat_output_subset"],
                    )
                    buf_loss = torch.mean(
                        torch.Tensor(
                            [
                                torch.nn.functional.mse_loss(
                                    buf_outputs[i, rel_subset], buf_logits[i]
                                )
                                for i, rel_subset in enumerate(
                                    buf_data["relative_output_subset"]
                                )
                            ]
                        )
                    )
                else:
                    buf_outputs = self.backbone(buf_data["examples"])
                    buf_loss = self.loss(buf_outputs, buf_data["targets"])

                loss += buf_loss

        self.gradient_update(loss)

        # Add relevant data to the buffer.
        buffer_data = {
            "examples": inputs,
            # 'examples': inputs if self.buffer.transform is None else non_aug_inputs,
            "targets": targets,
        }
        if self.training_mode == "open":
            # If necessary: Include indices of the classes/concepts (from the total list of classes/concepts)
            # used in the current task.
            buffer_data["output_subset"] = [
                self.backbone.output_subset for _ in range(len(outputs))
            ]
        self.buffer.add_data(**buffer_data)

        return loss.item()

    @property
    def checkpoint(self):
        return {"self": self.state_dict(), "buffer": self.buffer.checkpoint}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
        self.buffer.load_checkpoint(state_dict["buffer"])
