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

BASE_CLASSES = None
PRIMER = "{}"
CLASSES = None

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": None,
    "mean": (0.4671, 0.4444, 0.4057),
    "std": (0.244, 0.2393, 0.2419),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": True,
    "primer": PRIMER,
    "eval_only": True,
    "type": "retrieval",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "MSCOCO")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            os.makedirs(self.root, exist_ok=True)

            base_url = "https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/resolve/main/images_mscoco_2014_5k_test.zip?download=true"
            anno_url = "https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/resolve/main/test_5k_mscoco_2014.csv?download=true"
            _ = os.system(f"wget -O {self.root}/data.zip {base_url}")
            _ = os.system(f"wget -O {self.root}/annotations.csv {anno_url}")
            print("Extracting data...")
            _ = os.system(f"unzip -q {self.root}/data.zip -d {self.root}")
            print("Cleaning up...")
            _ = os.system(f"rm {self.root}/data.zip")
