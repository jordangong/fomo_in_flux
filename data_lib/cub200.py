import json
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/CUB200_classnames.json", "r"))
PRIMER = "A photo of a {}, a type of bird."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4856, 0.4994, 0.4324),
    "std": (0.2272, 0.2226, 0.2613),
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

        self.root = os.path.join(root, "CUB200")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        img_folder = f"{self.root}/cub-200-images-{self.split}"
        if not os.path.exists(img_folder):

            if (
                not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
                and self.download
            ):
                from onedrivedownloader import download

                print("Downloading dataset...")
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EZcsdvrGUnBHnkZNJOqNccMBFNqqEjSpjckiSePjmdqdNw?download=1"
                download(
                    ln,
                    filename=os.path.join(self.root, "cub_200_2011.zip"),
                    unzip=True,
                    unzip_path=self.root,
                    clean=True,
                )

            data_file = np.load(
                os.path.join(self.root, f"{self.split}_data.npz"), allow_pickle=True
            )
            data = np.array(data_file["data"])
            os.makedirs(img_folder, exist_ok=True)

            for i in tqdm.trange(len(data), desc="Converting CUB200 data to images..."):
                savepath = f"{img_folder}/{i}.png"
                Image.fromarray(data[i]).convert("RGB").save(savepath)
