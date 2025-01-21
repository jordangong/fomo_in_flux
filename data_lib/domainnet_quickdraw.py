import json
import os

import numpy as np
import pandas as pd
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(
    open("data_lib/00_info/DOMAINNET_QUICKDRAW_classnames.json", "r")
)
PRIMER = "A quickdraw of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.9525, 0.9525, 0.9525),
    "std": (0.194, 0.194, 0.194),
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

        self.root = os.path.join(root, "DOMAINNET_QUICKDRAW")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading dataset...")
            dataset_url = "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"
            train_split = "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt"
            test_split = "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt"
            torchvision.datasets.utils.download_and_extract_archive(
                dataset_url, download_root=self.root
            )
            os.system(f"wget -O {self.root}/quickdraw_train.txt {train_split}")
            os.system(f"wget -O {self.root}/quickdraw_test.txt {test_split}")