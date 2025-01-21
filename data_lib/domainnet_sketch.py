import json
import os

import numpy as np
import pandas as pd
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/DOMAINNET_SKETCH_classnames.json", "r"))
PRIMER = "A sketch of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.8314, 0.8258, 0.8168),
    "std": (0.188, 0.19, 0.19),
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

        self.root = os.path.join(root, "DOMAINNET_SKETCH")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        # Download data if needed.
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading dataset...")
            dataset_url = "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
            train_split = "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt"
            test_split = "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt"
            torchvision.datasets.utils.download_and_extract_archive(
                dataset_url, download_root=self.root
            )
            os.system(f"wget -O {self.root}/sketch_train.txt {train_split}")
            os.system(f"wget -O {self.root}/sketch_test.txt {test_split}")

        split = pd.read_csv(
            os.path.join(self.root, f"sketch_{self.split}.txt"),
            delimiter=" ",
            header=None,
        )
        self.data = list(split[0])
        class_to_idx = {
            key: i
            for i, key in enumerate(
                sorted(
                    np.unique([x.split("/")[1] for x in self.data]),
                    key=lambda x: x.lower(),
                )
            )
        }
        self.targets = []
        for x in self.data:
            self.targets.append(class_to_idx[x.split("/")[1]])
        self.data = [os.path.join(self.root, x) for x in self.data]
