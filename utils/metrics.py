# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, List

import numpy as np
import torch

AVAILABLE_METRICS = [
    "accuracy",
    "recall",
    "task_masked_accuracy",
    "dataset_masked_accuracy",
    "backward_transfer",
    "forgetting",
]


def give(metric):
    assert_str = (
        f"Metric {metric} not available. Please choose from {AVAILABLE_METRICS}."
    )
    assert metric in AVAILABLE_METRICS or any(
        [x in metric for x in AVAILABLE_METRICS]
    ), assert_str

    if "recall" in metric:
        return RecallAtK(name=metric, k=int(metric.split("@")[-1]))

    if metric == "accuracy":
        return Accuracy(name="accuracy")

    if metric == "task_masked_accuracy":
        return TaskMaskedAccuracy(name="task_masked_accuracy")

    if metric == "dataset_masked_accuracy":
        return DatasetMaskedAccuracy(name="dataset_masked_accuracy")

    if metric == "backward_transfer":
        return BackwardTransfer(name="backward_transfer")

    if metric == "forward_transfer":
        return ForwardTransfer(name="backward_transfer")

    if metric == "Forgetting":
        return Forgetting(name="forgetting")


class Metric:
    def __init__(self, name, stage):
        """
        Args:
            name: str, name of metric.
            stage: int, denotes if metric is computed for each task
                (0) or after all tasks (1).
        """
        self.name = name
        self.stage = stage

    def _can_be_applied(self, dataset_type: str):
        return dataset_type == "classification"

    def __call__(self, **kwargs):
        return {self.name: None}


class RecallAtK(Metric):
    def __init__(self, name="recall@1", k=1):
        super().__init__(name, stage=0)
        self.k = k

    def _can_be_applied(self, dataset_type: str):
        return dataset_type == "retrieval"

    def __call__(
        self, predictions: np.ndarray, targets: np.ndarray, device: None, **kwargs
    ):
        """
        Compute retrieval@k.

        Args:
            predictions: torch.Tensor of shape n_samples x n_retrieval_targets.
                Assumes that the retrieval targets are sorted such that for each sample at position i, the desired retrieval target is at position i,i.
            targets: torch.Tensor of shape n_samples.
        """
        # compute text-to-image retrieval
        if device is None:
            t2i_recall_idcs = np.argsort(predictions.T, axis=-1)[:, ::-1][:, : self.k]
            i2t_recall_idcs = np.argsort(predictions, axis=-1)[:, ::-1][:, : self.k]

        else:
            predictions = torch.from_numpy(predictions).to(device)
            t2i_recall_idcs = (
                torch.argsort(predictions.T, dim=-1, descending=True)[:, : self.k]
                .cpu()
                .numpy()
            )
            i2t_recall_idcs = (
                torch.argsort(predictions, dim=-1, descending=True)[:, : self.k]
                .cpu()
                .numpy()
            )

        t2i_recallatk = (
            np.mean(
                np.sum(t2i_recall_idcs == targets.astype(int).reshape(-1, 1), axis=-1)
                > 0
            )
            * 100
        )

        # Compute image-to-text retrieval
        a, b = np.unique(targets, return_counts=True)
        ranges = [0] + list(np.cumsum(b))
        corrects = []
        for i in range(len(a)):
            sample_targets = np.arange(*ranges[i : i + 2]).reshape(-1, 1)
            matches = i2t_recall_idcs[i : i + 1] == sample_targets
            corrects.append(matches.any())
        i2t_recallatk = np.mean(corrects) * 100

        # Return averaged recall score.
        avg_recallatk = (t2i_recallatk + i2t_recallatk) / 2

        return {self.name: avg_recallatk}


class Accuracy(Metric):
    def __init__(self, name="accuracy"):
        super().__init__(name, stage=0)

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs):
        """
        Compute accuracy.

        Args:
            predictions: torch.Tensor
        """
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.sum(predictions == targets) / len(targets) * 100
        return {self.name: accuracy}


class DatasetMaskedAccuracy(Metric):
    def __init__(self, name="dataset_masked_accuracy"):
        super().__init__(name, stage=0)

    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        dataset_idcs: List[int],
        **kwargs,
    ):
        """
        Compute dataset-masked accuracy - assuming all possible target values to be given
        in targets, drops the logit values for prediction entries not associated with the target
        range to negative infinity.
        Equivalent in function to TaskMaskedAccuracy.

        Args:
            predictions: np.ndarray
            targets: np.ndarray
        """
        predictions = copy.deepcopy(predictions)
        class_idcs_to_fix = sorted(
            list(set(range(predictions.shape[-1])) - set(dataset_idcs))
        )
        predictions[:, class_idcs_to_fix] = -float("inf")
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.sum(predictions == targets) / len(targets) * 100
        return {self.name: accuracy}


class TaskMaskedAccuracy(Metric):
    def __init__(self, name="task_masked_accuracy"):
        super().__init__(name, stage=0)

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs):
        """
        Compute task-masked accuracy - assuming all possible target values to be given
        in targets, drops the logit values for prediction entries not associated with the target
        range to negative infinity.

        Args:
            predictions: np.ndarray
            targets: np.ndarray
        """
        predictions = copy.deepcopy(predictions)
        unique_target_vals = np.unique(targets)
        class_idcs_to_fix = sorted(
            list(set(range(predictions.shape[-1])) - set(unique_target_vals))
        )
        predictions[:, class_idcs_to_fix] = -float("inf")
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.sum(predictions == targets) / len(targets) * 100
        return {self.name: accuracy}


class NotYetPossibleToComputeMetricException(Exception):
    pass


class BackwardTransfer(Metric):
    def __init__(self, name="backward_transfer"):
        super().__init__(name, stage=1)

    def __call__(
        self,
        current_task_scores: list[Any],
        past_task_scores: list[list] = None,
        **kwargs,
    ):
        """
        Computes backward transfer metric.

        Args:
            current_task_scores: Any output metric score
                associated with a current task.
            prev_task_scores: List of output metric scores
                from previous tasks.
        """
        n_tasks = len(current_task_scores)
        if n_tasks > 1 and past_task_scores is not None:
            scores = past_task_scores + [current_task_scores]
            li = list()
            for i in range(n_tasks - 1):
                li.append(scores[-1][i] - scores[i][i])
            bwt = np.mean(li)
            return {self.name: bwt}
        else:
            return {self.name: 0}


class ForwardTransfer(Metric):
    def __init__(self, name="forward_transfer"):
        super().__init__(name, stage=1)

    def __call__(
        self,
        current_task_scores: list[Any],
        random_scores: list[Any],
        past_task_scores: list[list] = None,
        **kwargs,
    ):
        """
        Computes forward transfer metric.

        Args:
            current_task_scores: Any output metric score
                associated with a current task.
            prev_task_scores: List of output metric scores
                from previous tasks.
            random_scores: output metric for a sequence of tasks generated
                by some random reference model.
        """
        n_tasks = len(current_task_scores)
        if n_tasks > 1 and past_task_scores is not None:
            scores = past_task_scores + [current_task_scores]
            li = list()
            for i in range(1, n_tasks):
                li.append(scores[i - 1][i] - random_scores[i])
            return {"forward_transfer": np.mean(li)}
        else:
            return {self.name: 0}


class Forgetting(Metric):
    def __init__(self, name="forgetting"):
        super().__init__(name, stage=1)

    def __call__(
        self,
        current_task_scores: list[Any],
        past_task_scores: list[list] = None,
        **kwargs,
    ):
        """
        Computes forgetting metric.

        Args:
            current_task_scores: Any output metric score
                associated with a current task.
            prev_task_scores: List of output metric scores
                from previous tasks.
        """
        n_tasks = len(current_task_scores)
        if n_tasks > 1 and past_task_scores is not None:
            scores = past_task_scores + [current_task_scores]
            li = list()
            for i in range(n_tasks - 1):
                scores[i] += [0.0] * (n_tasks - len(scores[i]))
            np_res = np.array(scores)
            maxx = np.max(np_res, axis=0)
            for i in range(n_tasks - 1):
                li.append(maxx[i] - scores[-1][i])
            return {self.name: np.mean(li)}
        else:
            return {self.name: 0}


def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return {"backward_transfer": np.mean(li)}


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return {"forward_transfer": np.mean(li)}


def forgetting(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)
