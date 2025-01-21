import json
import os
from typing import Tuple

import datasets
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/FSCOCO_classnames.json", "r"))
PRIMER = "A sketch of a {}"
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4671, 0.4444, 0.4057),
    "std": (0.244, 0.2393, 0.2419),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "retrieval",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "FSCOCO")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            os.makedirs(self.root, exist_ok=True)

            data_url = "http://cvssp.org/data/fscoco/fscoco.tar.gz"
            torchvision.datasets.utils.download_and_extract_archive(
                data_url, download_root=self.root
            )
