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

BASE_CLASSES = json.load(open("data_lib/00_info/PLACES365_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4577, 0.4412, 0.4079),
    "std": (0.2338, 0.2306, 0.2407),
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

        self.root = os.path.join(root, "PLACES365")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            os.makedirs(self.root, exist_ok=True)
            base_url = "http://data.csail.mit.edu/places/places365"
            # To ensure reasonable sizes, we use the test split for training & validation split for testing
            train_file = "train_256_places365standard.tar"
            val_file = "val_256.tar"
            label_file = "filelist_places365-standard.tar"

            print("Downloading training data...")
            os.system(f"wget -O {self.root}/{train_file} {base_url}/{train_file}")

            print("Downloading test data...")
            os.system(f"wget -O {self.root}/{val_file} {base_url}/{val_file}")

            print("Extracting training data...")
            os.system(f"tar -xf {self.root}/{train_file} -C {self.root}")
            os.system(f"mv {self.root}/{train_file} {self.root}/train")

            print("Extracting test data...")
            os.system(f"tar -xf {self.root}/{val_file} -C {self.root}")
            os.system(f'mv {self.root}/{val_file.split(".")[0]} {self.root}/val')

            os.system(f"wget -O {self.root}/file_info.tar {base_url}/{label_file}")
            os.system(f"tar -xf {self.root}/file_info.tar -C {self.root}")

        supp = "train_standard" if self.train else "val"
        info_file = np.array(pd.read_csv(f"{self.root}/places365_{supp}.txt"))

        if self.train:
            subset_folder = os.path.join(self.root, "PLACES365_subset_train")
            if not os.path.exists(subset_folder):
                print("Creating training subset...")
                os.makedirs(subset_folder, exist_ok=True)

                data, targets = [], []

                for x in info_file:
                    path, target = x[0].split(" ")
                    path = "/".join(path.split("/")[2:])
                    path = path.replace("/a/", "")
                    path = path.split("/")
                    if len(path) == 2:
                        path = "/".join(path)
                    else:
                        path = "/".join(path[:-2]) + f"-{path[-2]}" + f"/{path[-1]}"
                    target = int(target)
                    path = os.path.join(self.root, self.split, path)
                    data.append(path)
                    targets.append(target)
                # Create a ~7% (~120k) subset.
                data = data[::15]
                targets = targets[::15]

                for file in tqdm.tqdm(data, desc="Populating subset..."):
                    classname = file.split("/")[-2]
                    classfolder = os.path.join(subset_folder, classname)
                    os.makedirs(classfolder, exist_ok=True)
                    os.system(f"cp {file} {classfolder}")
