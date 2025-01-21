import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/QUILT_classnames.json", "r"))
PRIMER = "A histopathology photo depicting {} pathologies."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0, 0, 0),
    "std": (1, 1, 1),
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

        self.root = os.path.join(root, "QUILT")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            raise Exception(
                "Quilt1M data has do be manually downloaded from Zonodo including a download request, following https://quilt1m.github.io/."
            )
