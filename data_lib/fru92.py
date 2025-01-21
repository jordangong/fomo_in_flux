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

BASE_CLASSES = json.load(open("data_lib/00_info/FRU92_classnames.json", "r"))
PRIMER = "A photo of a {}, a type of fruit."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.6049, 0.5426, 0.4084),
    "std": (0.2244, 0.2357, 0.2556),
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

        self.root = os.path.join(root, "FRU92")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        veg200_root = self.root.replace("FRU92", "VEG200")
        assert_str = f"""
        There is currently no automatic dataset downloader for FRU92/VEG200. Please go to https://www.kaggle.com/datasets/zhaoyj688/vegfru,
        download manually and place the <archive.zip> inside either {self.root} or {veg200_root}.
        """
        assert os.path.isdir(self.root), assert_str
        os.makedirs(veg200_root, exist_ok=True)
        os.makedirs(os.path.join(veg200_root, "processed"), exist_ok=True)

        # Setting up datasets across FRU92 and VEG200.
        basefiles_path = os.path.join(self.root, "processed", "vegfru_list")
        if "processed" not in os.listdir(self.root):
            termcolor.cprint(
                "Note: Setting up FRU92 automatically sets up VEG200.",
                "yellow",
                attrs=["bold"],
            )
            assert_str = (
                f"Can not find <archive.zip> in either {self.root} or {veg200_root}"
            )
            assert "archive.zip" in os.listdir(
                self.root
            ) or "archive.zip" in os.listdir(veg200_root), assert_str

            if "archive.zip" in os.listdir(self.root):
                archive_file = os.path.join(self.root, "archive.zip")
            else:
                archive_file = os.path.join(veg200_root, "archive.zip")

            with zipfile.ZipFile(archive_file, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, "processed"))

            fru92_veg_path = os.path.join(self.root, "processed", "veg200_images")
            shutil.move(fru92_veg_path, os.path.join(veg200_root, "processed"))
            shutil.copytree(
                basefiles_path, os.path.join(veg200_root, "processed", "vegfru_list")
            )
