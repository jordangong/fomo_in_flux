import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/RETINOPATHY_classnames.json", "r"))
PRIMER = "A close-up light microscopy picture of {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4157, 0.2218, 0.0725),
    "std": (0.2416, 0.1338, 0.05),
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

        self.root = os.path.join(root, "RETINOPATHY")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            raise NotImplementedError(
                "Please manually download the retinopathy dataset from e.g. Kaggle: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data"
            )
