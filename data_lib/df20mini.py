import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/DF20MINI_classnames.json", "r"))
PRIMER = "A photo of the fungi {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.3682, 0.3808, 0.3434),
    "std": (0.1196, 0.1090, 0.1053),
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

        self.root = os.path.join(root, "DF20MINI")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if not os.path.exists(os.path.join(self.root, self.split)):

            if (
                not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
                and self.download
            ):
                print("Downloading dataset...")
                dataset_url = "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20M-images.tar.gz"
                torchvision.datasets.utils.download_and_extract_archive(
                    dataset_url, download_root=self.root
                )
                metadata_url = "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20M-metadata.zip"
                torchvision.datasets.utils.download_and_extract_archive(
                    metadata_url, download_root=self.root
                )

            if self.train:
                splits = pd.read_csv(
                    os.path.join(self.root, "DF20M-train_metadata_PROD.csv")
                )
            else:
                splits = pd.read_csv(
                    os.path.join(self.root, "DF20M-public_test_metadata_PROD.csv")
                )
            splits = splits.dropna(subset=["species"])

            data = [
                os.path.join(self.root, "DF20M", x) for x in list(splits["image_path"])
            ]
            targets = list(splits["species"])

            os.makedirs(os.path.join(self.root, self.split), exist_ok=True)
            for filepath, classname in tqdm.tqdm(
                zip(data, targets), total=len(data), desc="Moving data..."
            ):
                class_folder = os.path.join(self.root, self.split, classname)
                os.makedirs(class_folder, exist_ok=True)
                os.system(f'cp {filepath} "{class_folder}"')
