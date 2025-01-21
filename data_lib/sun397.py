import os
from typing import Tuple

import gdown
import json
import numpy as np
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/SUN397_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4729, 0.4570, 0.4210),
    "std": (0.2308, 0.2280, 0.2400),
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

        self.root = os.path.join(root, "SUN397")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            data_url = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
            torchvision.datasets.utils.download_and_extract_archive(
                data_url, download_root=self.root
            )
            partition_url = (
                "https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip"
            )
            torchvision.datasets.utils.download_and_extract_archive(
                partition_url, download_root=self.root
            )
            split_url = (
                "https://drive.google.com/uc?id=1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq"
            )
            gdown.download(
                split_url,
                os.path.join(self.root, "split_zhou_sun397.json"),
                quiet=False,
            )
            os.remove(os.path.join(self.root, "SUN397.tar.gz"))
            os.remove(os.path.join(self.root, "Partitions.zip"))

            split_dict = json.load(
                open(os.path.join(self.root, "split_zhou_sun397.json"), "r")
            )
            files = split_dict[self.split]
            data = [os.path.join(self.root, "SUN397", x[0]) for x in files]
            targets = [x[1] for x in files]

            if not os.path.exists(os.path.join(self.root, self.split)):
                os.makedirs(os.path.join(self.root, self.split), exist_ok=True)
                for filepath in tqdm.tqdm(data, total=len(data), desc="Moving data..."):
                    classname = filepath.split("/")[-4:-1]
                    if classname[0] == "SUN397":
                        classname = classname[-1]
                    else:
                        classname = "_".join(classname[1:])
                    class_folder = os.path.join(self.root, self.split, classname)
                    os.makedirs(class_folder, exist_ok=True)
                    os.system(f'cp {filepath} "{class_folder}"')
