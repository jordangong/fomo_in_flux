import json
import os
from typing import Tuple

import h5py
import itertools as it
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/ISICMELANOMA_classnames.json", "r"))
PRIMER = "A close-up photo of human skin with a mole exhibiting {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.8082, 0.6253, 0.5965),
    "std": (0.083, 0.096, 0.1075),
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

        self.root = os.path.join(root, "ISICMELANOMA")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("[WARNING] Now downloading large ISIC MELANOMA Dataset.")
            os.makedirs(self.root, exist_ok=True)
            data_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
            label_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv"

            print("Extracting...")
            os.system(f"wget -O {self.root}/train.zip {data_url}")
            os.system(f"wget -O {self.root}/labels.csv {label_url}")
            os.system(f"unzip -q {self.root}/train.zip")
