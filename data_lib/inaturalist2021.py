import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/iNATURALIST2021_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4642, 0.4810, 0.3768),
    "std": (0.1976, 0.1954, 0.1930),
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

        self.root = os.path.join(root, "iNATURALIST2021")
        self.PARAMS = PARAMS
        self.split = "train_mini" if self.train else "val"

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            os.makedirs(self.root, exist_ok=True)
            # To ensure reasonable sizes, we use the test split for training & validation split for testing
            base_url = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
            train_file = "train_mini.tar.gz"
            train_label_file = "train_mini.json.tar.gz"
            test_file = "val.tar.gz"
            test_label_file = "val.json.tar.gz"

            print("Downloading training data...")
            os.system(f"wget -O {self.root}/{train_file} {base_url}/{train_file}")
            os.system(
                f"wget -O {self.root}/{train_label_file} {base_url}/{train_label_file}"
            )

            print("Downloading test data...")
            os.system(f"wget -O {self.root}/{test_file} {base_url}/{test_file}")
            os.system(
                f"wget -O {self.root}/{test_label_file} {base_url}/{test_label_file}"
            )

            print("Extracting data...")
            os.system(f"tar xf {self.root}/{train_label_file} -C {self.root}")
            os.system(f"tar xf {self.root}/{train_file} -C {self.root}")
            os.system(f"mv {self.root}/{train_file} train")
            os.system(f"tar xf {self.root}/{test_label_file} -C {self.root}")
            os.system(f"tar xf {self.root}/{test_file} -C {self.root}")
            os.system(f"mv {self.root}/{test_file} train")

        data = []
        targets = []
        datapath = os.path.join(self.root, self.split)
        # We only use 25% of the classes.
        classes_to_use = sorted(os.listdir(datapath))[::4]
        for i, classname in enumerate(classes_to_use):
            classpath = os.path.join(datapath, classname)
            for file in os.listdir(classpath):
                data.append(os.path.join(classpath, file))
                targets.append(i)

        if not os.path.exists("data_lib/inaturalist2021_classes.json"):
            classnames = [" ".join(x.split("_")[1:]) for x in classes_to_use]
            json.dump(
                {"classnames": classnames},
                open("data_lib/inaturalist2021_classes.json", "w"),
            )

        self.data = data
        self.targets = targets
