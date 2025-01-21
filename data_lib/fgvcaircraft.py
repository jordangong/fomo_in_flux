import json
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/FGVCAircraft_classnames.json", "r"))
PRIMER = "A photo of a {}, a type of aircraft."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4818, 0.5122, 0.5356),
    "std": (0.1893, 0.1880, 0.2105),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": True,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "FGVCAircraft")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        self.fgvcaircraft_dataset = torchvision.datasets.FGVCAircraft(
            self.root,
            split=self.split,
            transform=None,
            target_transform=None,
            download=self.download,
        )
