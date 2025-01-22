import continual_lib

import torch


class Model(continual_lib.BaseContinualLearner):
    REQ_NON_AUG_INPUTS = False

    def __init__(
        self,
        args,
        backbone,
        head,
        loss,
        device,
        experiment,
        freeze_non_task_logits: bool = False,
        **kwargs
    ):
        super(Model, self).__init__(args, backbone, head, loss, device)

        self.freeze_non_task_logits = freeze_non_task_logits
        self.global_tasks_seen = []
        self.task_mask = torch.ones(experiment.total_num_classes)
        self.seen_targets = []
        self.global_mask_idcs = []

        # For probing, we only update the head parameters.
        self.backbone.module.freeze_features = True
        self.to_optimize = [{"params": self.backbone.module.head.parameters()}]

    def observe(self, inputs, targets, output_subset=None, **kwargs):
        """Continual Learner Single Training Step
        Args:
            inputs: [torch.Tensor: BS x C x W x H]
            targets: [torch.Tensor: BS (x 1)]
            output_subset:  [List/torch.Tensor/np.array] - denotes output logit subset to use for training in open-vocabulary continual learning.

        """
        self.opt.zero_grad()
        global_task = kwargs["experiment"].global_task
        with torch.amp.autocast("cuda"):
            if (
                self.freeze_non_task_logits
                and global_task not in self.global_tasks_seen
            ):
                if global_task not in self.global_tasks_seen:
                    total_num_classes = kwargs["experiment"].total_num_classes
                    self.task_mask = torch.ones(total_num_classes)
                    warm_logit_idcs = kwargs[
                        "experiment"
                    ].give_class_indices_of_current_task_targets()
                    self.task_mask[warm_logit_idcs] = 0
                    # self.seen_targets.extend(list(warm_logit_idcs))
                    # self.seen_targets = sorted(list(set(self.seen_targets)))
                    # self.global_mask_idcs = sorted(list(set(range(total_num_classes)) - set(self.seen_targets)))
                    self.global_tasks_seen.append(global_task)

            outputs = self.backbone(inputs)
            if self.freeze_non_task_logits:
                outputs[:, torch.where(self.task_mask)[0]] = -float("inf")
            loss = self.loss(outputs, targets)

        self.gradient_update(loss)

        return loss.item()

    @property
    def checkpoint(self):
        return {"self": self.state_dict()}

    def load_from_checkpoint(self, state_dict):
        self.load_state_dict(state_dict["self"])
