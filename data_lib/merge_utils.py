import importlib
import os
from typing import List

from omegaconf import DictConfig, open_dict
import torch
import torchvision


# Ensure that both samplers simply provide the right datastream without any additional changes needed.
class UniformSampler(torch.utils.data.Dataset):
    def __init__(
        self,
        args: DictConfig,
        root: str,
        dataset_list: List[torch.utils.data.Dataset],
        download: bool = True,
        samples_per_dataset: int = 1,
    ) -> None:
        self.args = args
        self.root = root
        self.dataset_list = dataset_list
        self.total_len = sum([len(x) for x in self.dataset_list])

    def __len__(self):
        return self.total_len


class UniformChunker(torch.utils.data.Dataset):
    def __init__(
        self, root: str, dataset_list: list, train: bool = True, download: bool = True
    ) -> None:
        pass

    def __len__(self):
        return len(self.data)

    def get_transform(self):
        return torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(), self.transform]
        )

    def set_transforms(self, transform: torchvision.transforms = None):
        if transform is None:
            if self.train:
                self.transform = torchvision.transforms.Compose(
                    self.PARAMS["train_transforms"]
                )
            else:
                self.transform = torchvision.transforms.Compose(
                    self.PARAMS["test_transforms"]
                )
        else:
            self.transform = torchvision.transforms.Compose(transform)
        self.target_transform = None


class MergedEvaluator:
    def __init__(self):
        pass
