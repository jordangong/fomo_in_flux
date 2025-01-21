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

BASE_CLASSES = json.load(open("data_lib/00_info/OPENIMAGES_classnames.json", "r"))
PRIMER = "A photo of a {}."
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
    "create_resized_variant_if_possible": True,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}

###################### Instructions to download the openimages dataset manually ######################
##
## 1. Download the downloader script using `wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py`
## 2. Download the relevant image indices from here: `https://drive.google.com/file/d/1PgmSe-rOKCMbyyUVDgB1qaASL4zZ-Jxs/view`
## 3. Create dir: `mkdir <data_root/OPENIMAGES>`
## 4. Run `python downloader.py download_test_image_indices.txt --download_folder <data_root>/OPENIMAGES/images`
##
#####################################################################################################


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "OPENIMAGES")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        assert_str = f"""
        There is currently no automatic dataset downloader for OPENIMAGES. Please see the above instructions to download the dataset manually.
        """
        assert os.path.isdir(self.root), assert_str

        if os.path.isdir(os.path.join(self.root, "images")):
            return

        assert os.path.isdir(
            self.root
        ), "OpenImages data is not setup correctly, please delete and try again."
        assert os.path.isdir(
            os.path.join(self.root, "images")
        ), "OpenImages data is not setup correctly, please delete and try again."
        assert (
            len(os.listdir(os.path.join(self.root, "images"))) == 123926
        ), "OpenImages data is not setup correctly, please delete and try again."
