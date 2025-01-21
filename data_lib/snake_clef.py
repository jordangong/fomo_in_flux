import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/SNAKECLEF_classnames.json", "r"))
PRIMER = "A photo of a {} snake."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5437, 0.5126, 0.4661),
    "std": (0.2237, 0.2272, 0.2367),
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

        self.root = os.path.join(root, "SNAKECLEF")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            os.makedirs(self.root, exist_ok=True)
            train_url = "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-train-medium_size.tar.gz"
            val_url = "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-val-medium_size.tar.gz"
            torchvision.datasets.utils.download_and_extract_archive(
                train_url, download_root=self.root, remove_finished=True
            )
            torchvision.datasets.utils.download_and_extract_archive(
                val_url, download_root=self.root, remove_finished=True
            )
            train_metadata = "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-TrainMetadata-iNat.csv"
            val_metadata = "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-ValMetadata.csv"
            os.system(f"wget -O {self.root}/train.csv {train_metadata}")
            os.system(f"wget -O {self.root}/val.csv {val_metadata}")
