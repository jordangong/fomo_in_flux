import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import shutil
import termcolor
import torchvision.transforms
import tqdm
import zipfile

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/CLRS_classnames.json", "r"))
PRIMER = "A satellite image of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    ### Note: using default ImageNet mean and std -- does not matter for continual pretraining as overwritten by pretrained model mean and stds
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
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

        self.root = os.path.join(root, "CLRS")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        assert_str = f"""
        There is currently no automatic dataset downloader for CLRS. Please go to https://drive.google.com/file/d/1hOQHIxF2NfXyDjKf3hUHf5Jw7qjq-nJv/view,
        download manually and place the <CLRS.zip> inside {self.root}.
        """
        assert os.path.isdir(self.root), assert_str

        if os.path.isdir(os.path.join(self.root, "CLRS")) and os.path.isdir(
            os.path.join(self.root, "CLRS", "CLRS")
        ):
            return

        assert_str = f"Can not find <CLRS.zip> in {self.root}"
        assert "CLRS.zip" in os.listdir(self.root), assert_str

        archive_file = os.path.join(self.root, "CLRS.zip")

        with zipfile.ZipFile(archive_file, "r") as zip_ref:
            zip_ref.extractall(self.root)

        assert os.path.isdir(
            self.root
        ), "CLRS data is not setup correctly, please delete and try again."
        assert os.path.isdir(
            os.path.join(self.root, "CLRS")
        ), "CLRS data is not setup correctly, please delete and try again."
        assert os.path.isdir(
            os.path.join(self.root, "CLRS", "CLRS")
        ), "CLRS data is not setup correctly, please delete and try again."
        assert (
            len(os.listdir(os.path.join(self.root, "CLRS", "CLRS"))) == 25
        ), "CLRS data is not setup correctly, please delete and try again."
