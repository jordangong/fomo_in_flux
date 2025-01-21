import json
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/GTSRB_classnames.json", "r"))
PRIMER = 'A centered photo of a "{}" traffic sign.'
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.3415, 0.3124, 0.3214),
    "std": (0.1663, 0.1662, 0.1762),
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

        self.root = os.path.join(root, "GTSRB")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading dataset...")
            base_url = (
                "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
            )
            torchvision.datasets.utils.download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=self.root,
                md5="513f3c79a4c5141765e10e952eaa2478",
            )