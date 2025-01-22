import numpy as np
import torch
import torchvision
import tqdm

import continual_lib
from continual_lib.utils import buffer


class QuickIterator(torch.utils.data.Dataset):
    def __init__(self, samples, targets, transform):
        self.samples = samples
        self.targets = targets
        self.transform = transform
        if self.transform is not None:
            if isinstance(self.samples[0], torch.Tensor):
                self.transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToPILImage(), self.transform]
                )
        else:
            self.transform = lambda x: x

    def __getitem__(self, idx):
        return self.transform(self.samples[idx]), self.targets[idx]

    def __len__(self) -> None:
        return len(self.samples)


def fit_buffer(
    model, f_loss, buffer, buffer_transform, device, args, epochs=50, cos_lr_multi=0.01
) -> None:
    base_lr = args.experiment.optimizer.lr
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=args.experiment.optimizer.momentum,
        weight_decay=args.experiment.optimizer.weight_decay,
        nesterov=args.experiment.optimizer.nesterov_momentum,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=base_lr * cos_lr_multi
    )

    quick_iter_dataset = QuickIterator(
        buffer.examples, buffer.targets, buffer_transform
    )
    data_iterator = torch.utils.data.DataLoader(
        quick_iter_dataset,
        num_workers=args.experiment.dataset.num_workers,
        shuffle=True,
        batch_size=args.experiment.task.batch_size,
    )
    epoch_iterator = tqdm.tqdm(range(epochs), desc="Fitting to GDumb Buffer")

    scaler = torch.amp.GradScaler('cuda')

    for epoch in epoch_iterator:

        if epoch <= 0:  # Warm start of 1 epoch
            ref_lr = base_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = ref_lr
        elif epoch == 1:  # Then set to maxlr
            ref_lr = base_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr
        else:
            scheduler.step()
            ref_lr = scheduler.get_last_lr()

        epoch_iterator.set_description_str(
            f"Fitting to GDumb Buffer with lr = {ref_lr}"
        )

        is_training = model.training
        _ = model.train()

        accs = []
        losses = []

        for buf_inputs, buf_targets in data_iterator:
            optimizer.zero_grad()

            buf_inputs = buf_inputs.to(device)
            buf_targets = buf_targets.to(device)

            with torch.amp.autocast("cuda"):
                buf_outputs = model(buf_inputs)
                loss = f_loss(buf_outputs, buf_targets.to(torch.long))
                scaler.scale(loss).backward()
                if args.experiment.optimizer.clip_grad_norm > 0:
                    _ = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.experiment.optimizer.clip_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()

            accs.append(
                torch.sum(torch.argmax(buf_outputs, dim=-1) == buf_targets).item()
                / len(buf_outputs)
                * 100
            )
            losses.append(loss.item())

        epoch_iterator.set_postfix_str(
            "Loss: {0:3.6f} | Acc: {1:3.2f}%".format(np.mean(losses), np.mean(accs))
        )

        if not is_training:
            _ = model.eval()


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = True
    ON_TASK_LEARNING = False

    def __init__(
        self,
        args,
        backbone,
        head,
        loss,
        device,
        experiment,
        fitting_epochs,
        cos_lr_multi,
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)
        self.buffer = buffer.Buffer(
            args.experiment.buffer.size,
            args.experiment.buffer.batch_size,
            device,
            (
                None
                if not args.experiment.buffer.with_transform
                else experiment.train_transform
            ),
            training_mode=self.training_mode,
        )
        self.fitting_epochs = fitting_epochs
        self.cos_lr_multi = cos_lr_multi

    def observe(self, inputs, targets, non_aug_inputs=None, **kwargs):
        buffer_data = {
            "examples": inputs if non_aug_inputs is None else non_aug_inputs,
            "targets": targets,
        }
        self.buffer.add_data(**buffer_data)
        return None

    def end_task(self, experiment, **kwargs):
        # print(experiment.global_taks, experiment.total_num_tasks)
        if experiment.global_task < experiment.total_num_tasks - 1:
            return

        print("\nStarting GDumb Buffer Fitting!\n")

        fit_buffer(
            self.backbone,
            self.loss,
            self.buffer,
            self.buffer.transform,
            self.device,
            self.args,
            self.fitting_epochs,
            self.cos_lr_multi,
        )

    def end_observe(self) -> None:
        pass

    @property
    def checkpoint(self):
        return {"self": self.state_dict(), "buffer": self.buffer.checkpoint}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
        self.buffer.load_checkpoint(state_dict["buffer"])
