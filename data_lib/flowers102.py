import json
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/FLOWERS102_classnames.json", "r"))
PRIMER = "A photo of a {}, a type of flower."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4329, 0.3819, 0.2964),
    "std": (0.2610, 0.2120, 0.2236),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "FLOWERS102")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        # NOTE: To provide enough training samples, we swap train & test splits for FLOWERS102
        self.flowers102_dataset = torchvision.datasets.Flowers102(
            self.root,
            split="test" if self.train else "train",
            transform=None,
            target_transform=None,
            download=self.download,
        )
