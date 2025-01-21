# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import pathlib
import shutil
import sys
from typing import Union, List

import argparse
from utils.metrics import *
from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np

useless_args = [
    "dataset",
    "tensorboard",
    "validation",
    "model",
    "csv_log",
    "notes",
    "load_best_args",
]


def print_mean_accuracy(
    mean_acc: np.ndarray, task_number: int, setting: str, header: str = ""
) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    header = f"[{header}] " if len(header) else ""
    if setting == "domain-il":
        mean_acc, _ = mean_acc
        print(
            "{}Accuracy for {} task(s): {} %".format(
                header, task_number, round(mean_acc, 2)
            ),
            file=sys.stderr,
        )
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print(
            "{}Accuracy for {} task(s): \t [Class-IL]: {} %"
            " \t [Task-IL]: {} %".format(
                header,
                task_number,
                round(mean_acc_class_il, 2),
                round(mean_acc_task_il, 2),
            ),
            file=sys.stderr,
        )


def wandb_logger(
    mean_acc: Union[np.ndarray, dict], task_number: int, setting: str
) -> None:
    """
    Logs the mean accuracy to Weights & Biases.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    import wandb

    if not isinstance(mean_acc, dict):
        mean_acc = {"default": mean_acc}

    if setting == "domain-il":
        log_dict = {}
        for key, acc in mean_acc.items():
            log_dict[f"{key}_accuracy"] = acc[0]
    else:
        log_dict = {}
        for key, acc in mean_acc.items():
            log_dict[f"{key}_class_il_accuracy"] = acc[0]
            log_dict[f"{key}_task_il_accuracy"] = acc[1]
    log_dict["task_number"] = task_number
    wandb.log(log_dict)


class ResultManager:
    def __init__(
        self,
        project: str,
        group: str,
        seed: int,
        run_settings: list,
        chkpt_folder: str,
        num_tasks: List[int],
        log_to: list = ["csv", "wandb"],
        wandb_key: str = "",
        no_checkpoint: bool = False,
        args: argparse.Namespace = None,
    ):

        self.log_to = log_to
        self.num_tasks = num_tasks

        # Project details.
        self.project = project
        self.group = group
        self.seed = seed
        self.run_settings = run_settings
        self.setting = run_settings[0]

        # Offline log & checkpointing setup.
        self.no_checkpoint = no_checkpoint
        self.chkpt_folder = chkpt_folder
        self.run_name = f"{self.group}_s-{self.seed}"
        self.path = pathlib.Path(chkpt_folder) / self.project
        for setting in self.run_settings:
            self.path /= setting
        self.path /= self.run_name
        if self.no_checkpoint and os.path.exists(self.path):
            print(
                f"\n-> Found folder at {self.path}, removing it [no_checkpoint==True]."
            )
            shutil.rmtree(self.path)
            os.makedirs(self.path, exist_ok=True)
        else:
            os.makedirs(self.path, exist_ok=True)

        # Weights & Biases logging setup.
        self.wandb_key = wandb_key
        if "wandb" in self.log_to:
            self._init_wandb(args)

        # Metrics to log.
        self.results = {}
        self.stage_counts = {}

    def _init_wandb(self, args):
        import wandb

        os.environ["WANDB_API_KEY"] = self.wandb_key
        wandb.init(
            project=self.project,
            group=self.group,
            name=self.run_name,
            dir="wandb",
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    def collect_metrics(
        self,
        metrics: dict,
        names: list,
        parent: str = "train",
        surpress_stage_count: bool = False,
    ):
        assert_str = (
            f"Provided metrics do not align with provided metric names in count."
        )
        assert len(metrics) == len(names), assert_str
        if parent not in self.results:
            self.results[parent] = {}
            self.stage_counts[parent] = 0

        for i, name in enumerate(names):
            if name not in self.results[parent]:
                self.results[parent][name] = []
            self.results[parent][name].append(metrics[i])

        if not surpress_stage_count:
            self.stage_counts[parent] += 1

    def write_results(self):
        for parent, results in self.results.items():
            for i in range(1, self.stage_counts[parent] + 1)[::-1]:
                log_dict = {}
                for name, result_list in results.items():
                    res = result_list[-i]
                    if isinstance(result_list[-i], list):
                        log_dict[f"mean_{name}"] = np.round(np.mean(res), 4)
                        for t in range(self.num_tasks):
                            if t < len(res):
                                log_dict[f"task-{t+1}_{name}"] = np.round(res[t], 4)
                            else:
                                log_dict[f"task-{t+1}_{name}"] = 0
                    else:
                        log_dict[name] = np.round(res, 4)

                if "wandb" in self.log_to:
                    import wandb

                    wandb_log_dict = {}
                    for name, item in log_dict.items():
                        if name != "task_number":
                            wandb_log_dict[f"{parent}.{name}"] = item
                        else:
                            wandb_log_dict[name] = item
                    wandb.log()

                if "csv" in self.log_to:
                    if not os.path.exists(self.path / f"{parent}_metrics.csv"):
                        with open(self.path / f"{parent}_metrics.csv", "w") as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow(list(log_dict.keys()))
                    with open(self.path / f"{parent}_metrics.csv", "a") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(list(log_dict.values()))
                self.stage_counts[parent] = 0

    @property
    def checkpoint(self):
        return {"results": self.results, "stage_counts": self.stage_counts}

    def load_from_checkpoint(self, checkpoint):
        self.results = checkpoint["results"]
        self.stage_counts = checkpoint["stage_counts"]
