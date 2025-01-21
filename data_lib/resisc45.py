import json
import os
from typing import Tuple

import gdown
import numpy as np
import pandas as pd
import patoolib
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/RESISC45_classnames.json", "r"))
PRIMER = "A satellite image of {}."
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
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "RESISC45")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading dataset...")
            os.makedirs(self.root, exist_ok=True)
            gdown.download(
                "https://drive.google.com/uc?id=1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv",
                os.path.join(self.root, "base_file.rar"),
                quiet=False,
            )
            gdown.download(
                "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",
                os.path.join(self.root, "resisc45-train.txt"),
                quiet=False,
            )
            gdown.download(
                "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",
                os.path.join(self.root, "resisc45-test.txt"),
                quiet=False,
            )
            # gdown.download("https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt", os.path.join(self.root, 'resisc45-val.txt'), quiet=False)

            torchvision.datasets.utils.download_and_extract_archive(
                "https://figshare.com/ndownloader/files/30871912",
                download_root=self.root,
            )
            # If this throws a No-rar-library error, simply install via sudo apt install rar.
            try:
                patoolib.extract_archive(
                    os.path.join(self.root, "base_file.rar"), outdir=self.root
                )
            except:
                raise NotImplementedError(
                    "Do rar-library installed on your system. Please do so via e.g. <sudo apt install rar>!"
                )
            os.remove(os.path.join(self.root, "base_file.rar"))
