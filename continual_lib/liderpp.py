import numpy as np
import torch

import continual_lib
from continual_lib.utils import buffer


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(self, args, backbone, head, loss, device, experiment, alpha):
        super(Model, self).__init__(args, backbone, head, loss, device)
        self.buffer = buffer.Buffer(
            args.experiment.buffer.size,
            args.experiment.buffer.batch_size,
            device,
            None,
            training_mode=self.training_mode,
        )
        # self.buffer = buffer.Buffer(
        #     args.experiment.buffer.size,  args.experiment.buffer.batch_size,
        #     device, None if not args.experiment.buffer.with_transform else experiment.train_transform,
        #     training_mode=self.training_mode
        # )
        self.alpha = alpha

    def observe(self, inputs, targets, non_aug_inputs=None, **kwargs):
        self.opt.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = self.backbone(inputs)
            loss = self.loss(outputs, targets)

            if not self.buffer.is_empty():
                buf_out = self.buffer.get_data(**kwargs)
                buf_inputs, buf_logits = buf_out[:2]

                if self.training_mode == "open":
                    # For open-class continual learning, i.e. we only use classes we have seen before,
                    # we get output logits used across buffer outputs.
                    # TODO: Move usage to buffer class!
                    per_sample_output_subset = buf_out[-1]
                    output_subset = np.array(
                        sorted(
                            list(set([x for y in per_sample_output_subset for x in y]))
                        )
                    )
                    idx_to_rel_idx = {idx: i for i, idx in enumerate(output_subset)}
                    relative_output_subset = [
                        [idx_to_rel_idx[x] for x in y] for y in per_sample_output_subset
                    ]
                    buf_outputs = self.backbone(buf_inputs, output_subset=output_subset)
                    buf_loss = torch.mean(
                        torch.Tensor(
                            [
                                torch.nn.functional.mse_loss(
                                    buf_outputs[i, rel_subset], buf_logits[i]
                                )
                                for i, rel_subset in enumerate(relative_output_subset)
                            ]
                        )
                    )
                else:
                    buf_outputs = self.backbone(buf_inputs)
                    buf_loss = torch.nn.functional.mse_loss(buf_outputs, buf_logits)
                loss += self.alpha * buf_loss

        self.gradient_update(loss)

        # Add relevant data to the buffer.
        buffer_data = {
            "examples": inputs,
            # 'examples': inputs if self.buffer.transform is None else non_aug_inputs,
            "logits": outputs.data,
        }
        if self.training_mode == "open":
            # If necessary: Include indices of the classes/concepts (from the total list of classes/concepts)
            # used in the current task.
            output_subset = self.backbone.output_subset
            buffer_data["output_subset"] = [output_subset for _ in range(len(outputs))]
        self.buffer.add_data(**buffer_data)

        return loss.item()

    @property
    def checkpoint(self):
        return {"self": self.state_dict(), "buffer": self.buffer.checkpoint}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
        self.buffer.load_checkpoint(state_dict["buffer"])
