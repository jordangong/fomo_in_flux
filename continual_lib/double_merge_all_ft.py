import copy
from continual_lib.double_merge_base import DoubleMergeBase


class Model(DoubleMergeBase):
    """All-merge for training, last checkpoint for evaluation."""

    def define_training_weights(self, **kwargs):
        for mode in ["backbone", "head"]:
            if len(self.checkpoint_storage["running"][mode]) > 1:
                self.f_merge[mode].set_weight_coefficients(None)
                checkpoints = self.checkpoint_storage["running"][mode][1:]
                self.checkpoint_storage["train"][mode] = copy.deepcopy(
                    self.all_merge(checkpoints, mode)
                )
            else:
                self.checkpoint_storage["train"][mode] = copy.deepcopy(
                    self.checkpoint_storage["running"][mode][-1]
                )

    def define_evaluation_weights(self, **kwargs):
        for mode in ["backbone", "head"]:
            self.checkpoint_storage["eval"][mode] = copy.deepcopy(
                self.checkpoint_storage["running"][mode][-1]
            )
