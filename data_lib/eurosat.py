import json
import os
import ssl
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/EuroSAT_classnames.json", "r"))
PRIMER = "A centered satellite photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.3444, 0.3803, 0.4078),
    "std": (0.0931, 0.0648, 0.0542),
    # Default Imagesize
    "img_size": 64,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "EuroSAT")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        self.eurosat_dataset = torchvision.datasets.EuroSAT(
            self.root, transform=None, target_transform=None, download=self.download
        )
