import json
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/COUNTRY211_classnames.json", "r"))
PRIMER = "A photo representing the country of {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4569, 0.4503, 0.4208),
    "std": (0.2357, 0.2304, 0.2405),
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

        self.root = os.path.join(root, "COUNTRY211")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading Dataset...")
            dataset_url = "https://openaipublic.azureedge.net/clip/data/country211.tgz"
            torchvision.datasets.utils.download_and_extract_archive(
                dataset_url, download_root=self.root
            )
