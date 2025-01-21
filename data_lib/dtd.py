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

BASE_CLASSES = json.load(open("data_lib/00_info/DTD_classnames.json", "r"))
PRIMER = "A photo of a {} texture."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5288, 0.4730, 0.4247),
    "std": (0.1757, 0.1765, 0.1723),
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

        self.root = os.path.join(root, "DTD")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        self.dtd_dataset = torchvision.datasets.DTD(
            self.root,
            split=self.split,
            transform=None,
            target_transform=None,
            download=self.download,
        )
