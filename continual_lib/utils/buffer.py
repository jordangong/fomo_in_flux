# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Tuple, List, Union

import numpy as np
import torch
import torchvision

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, transform, length):
        self.examples = []
        self.logits = []
        self.transform = (
            torchvision.transforms.Compose(
                [torchvision.transforms.ToPILImage(), transform]
            )
            if transform is not None
            else lambda x: x
        )
        self.length = length

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.examples))
        return self.transform(self.examples[idx]), self.logits[idx]

    def __len__(self):
        return 100000000000000000000


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        device: torch.device,
        training_mode: str,
        transform: torchvision.transforms = None,
        n_tasks=None,
        buffer_mode="reservoir",
        **kwargs
    ):
        assert buffer_mode in ["ring", "reservoir"]
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.transform = transform
        self.transform_adjusted = False
        self.num_seen_examples = 0
        self.functional_index = eval(buffer_mode)
        self.training_mode = training_mode
        if buffer_mode == "ring":
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = self.buffer_size // n_tasks

        self.attributes = None

    def checkpoint(self):
        chkpt_dict = {"num_seen_examples": self.num_seen_examples}
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self.attr_str)
                if isinstance(attr[0], torch.Tensor):
                    chkpt_dict[attr_str] = [x.detach().cpu() for x in attr]
                else:
                    chkpt_dict[attr_str] = attr
        return chkpt_dict

    def load_checkpoint(self, state_dict):
        self.num_seen_examples = state_dict["num_seen_examples"]
        for attr_str in [key for key in state_dict.keys() if key in self.attributes]:
            attr = state_dict[attr]
            if isinstance(attr[0], torch.Tensor):
                setattr(self, attr_str, [x.to(self.device) for x in attr])
            else:
                setattr(self, attr_str, attr)

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                if isinstance(attr[0], torch.Tensor):
                    setattr(self, attr_str, [x.to(device) for x in attr])
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_buffer_elements(self, attribute_dict) -> None:
        self.attributes = list(attribute_dict.keys())
        for attr_str in self.attributes:
            # attr_buffer = [None for _ in range(self.buffer_size)]
            item = attribute_dict[attr_str]
            dtype = (
                type(item[0]) if not isinstance(item[0], np.ndarray) else item[0].dtype
            )
            shape = [] if not isinstance(item[0], np.ndarray) else list(item[0].shape)
            attr_buffer = np.empty([self.buffer_size, *shape], dtype=dtype)
            setattr(self, attr_str, attr_buffer)

    def add_data(self, **kwargs):
        kwargs = {
            key: item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
            for key, item in kwargs.items()
        }

        if self.attributes is None:
            self.init_buffer_elements(kwargs)

        resevoir_indices = []
        num_additions = np.max(
            [
                len(kwargs[attribute])
                for attribute in self.attributes
                if attribute in kwargs
            ]
        )
        for i in range(num_additions):
            resevoir_indices.append(reservoir(self.num_seen_examples, self.buffer_size))
            self.num_seen_examples += 1

        for attr_str in self.attributes:
            attr = kwargs[attr_str]
            if attr is not None:
                for i, index in enumerate(resevoir_indices):
                    if index >= 0:
                        getattr(self, attr_str)[index] = attr[i]

    def get_data(
        self, size: int = None, transform: torchvision.transforms = None, **kwargs
    ) -> Tuple:
        if self.attributes is None:
            return None

        if size is None:
            size = self.batch_size

        if size > min(self.num_seen_examples, self.buffer_size):
            size = min(self.num_seen_examples, self.buffer_size)

        choice = np.random.choice(
            min(self.num_seen_examples, self.buffer_size), size=size, replace=False
        )

        if (
            transform is None
            and self.transform is not None
            and "images" in self.attributes
        ):
            if isinstance(self.images[choice[0]], torch.Tensor):
                transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToPILImage(), self.transform]
                )
            else:
                transform = self.transform

        if transform is None:
            transform = lambda x: x

        return_data = {}

        if "images" in self.attributes:
            return_data["images"] = torch.stack(
                [transform(torch.from_numpy(self.images[i])) for i in choice]
            ).to(self.device)

        for attr_str in self.attributes:
            if attr_str not in ["images"]:
                attr = getattr(self, attr_str)
                if attr[0] is not None:
                    attr_val = attr[choice]
                    if not isinstance(attr_val[0], str):
                        attr_val = torch.from_numpy(attr_val).to(self.device)
                    return_data[attr_str] = attr_val

        return_data["selected_indices"] = torch.tensor(choice).to(self.device)

        # For contrastive training, the buffer logits have to be recomputed based on the selected samples.
        if (
            self.training_mode == "contrastive"
            and "text_features" in return_data
            and "features" in return_data
        ):
            return_data["logits"] = (
                torch.nn.functional.normalize(return_data["features"], dim=-1)
                @ torch.nn.functional.normalize(return_data["text_features"], dim=-1).T
            )

        return return_data

    def get_data_by_index(
        self, indexes, transform: torchvision.transforms = None
    ) -> Tuple:
        if self.attributes is None:
            return None

        if transform is None:
            transform = self.transform

        if transform is None:
            self.transform = lambda x: x

        return_data = {}
        if "images" in self.attributes:
            return_data["images"] = (
                torch.stack(
                    [self.transform(ee.cpu()) for ee in self.images[indexes]]
                ).to(self.device),
            )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                return_data[attr_str] = attr[indexes]

        return return_data

    def is_empty(self) -> bool:
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: torchvision.transforms = None) -> Tuple:
        if self.attributes is None:
            return None

        if transform is None:
            transform = self.transform

        if transform is None:
            transform = lambda x: x

        return_data = {}

        if "images" in self.attributes:
            return_data["images"] = torch.stack(
                [transform(self.images[i]) for i in range(len(self.images))]
            ).to(self.device)

        for attr_str in self.attributes[1:]:
            # We get items from the buffer that we have in storage.
            attr = getattr(self, attr_str)
            if attr[0] is not None:
                if isinstance(attr[0], torch.Tensor):
                    attr = torch.stack(attr).to(self.device)
                return_data[attr_str] = attr

        return return_data

    def empty(self) -> None:
        if self.attributes is not None:
            for attr_str in self.attributes:
                if hasattr(self, attr_str):
                    delattr(self, attr_str)
            self.num_seen_examples = 0
            self.attributes = None
