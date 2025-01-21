import os
from typing import Tuple

import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/ZAPPOS50k_classnames.json", "r"))
PRIMER = "A photo of {} shoes."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.7813, 0.7533, 0.7456),
    "std": (0.2932, 0.3233, 0.3303),
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

        self.root = os.path.join(root, "ZAPPOS50k")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            os.makedirs(self.root, exist_ok=True)
            base_url = "https://vision.cs.utexas.edu/projects/finegrained/utzap50k"
            # To ensure reasonable sizes, we use the test split for training & validation split for testing
            train_file = "ut-zap50k-images.zip"

            print("Downloading training data...")
            os.system(f"wget -O {self.root}/{train_file} {base_url}/{train_file}")
            os.system(f"unzip -q {self.root}/{train_file} {self.root}")

            os.makedirs(f"{self.root}/data", exist_ok=True)
            base_folder = os.path.join(self.root, "ut-zap50k-images")
            for folder in sorted(os.listdir(base_folder)):
                for subclass in sorted(os.listdir(os.path.join(base_folder, folder))):
                    for subsubclass in sorted(
                        os.listdir(os.path.join(base_folder, folder, subclass))
                    ):
                        source_folder = os.path.join(
                            base_folder, folder, subclass, subsubclass
                        )
                        target_folder = os.path.join(
                            self.root, "data", f"{folder}_{subclass}_{subsubclass}"
                        )
                        os.makedirs(target_folder, exist_ok=True)
                        os.system(f'cp -r "{source_folder}"/* "{target_folder}"')

            os.system(f"rm -r {self.root}/ut-zap50k-images")
