import json
import os
from typing import Tuple

from PIL import Image
import numpy as np
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/FashionMNIST_classnames.json", "r"))
PRIMER = "A black-and-white photo of a {}, a type of clothing."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.2859, 0.2859, 0.2859),
    "std": (0.3530, 0.3530, 0.3530),
    # Default Imagesize
    "img_size": 28,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "FashionMNIST")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        image_folder = f"{self.root}/fashionmnist-images-{self.split}"
        if not os.path.exists(image_folder):
            self.fashionmnist_dataset = torchvision.datasets.FashionMNIST(
                self.root,
                self.train,
                transform=None,
                target_transform=None,
                download=self.download,
            )

            data = np.array(self.fashionmnist_dataset.data)
            targets = np.array(self.fashionmnist_dataset.targets)
            os.makedirs(image_folder, exist_ok=True)
            for i in tqdm.trange(len(data), desc="Creating FashionMNIST images..."):
                savepath = f"{image_folder}/{targets[i]}-{i}.png"
                _ = Image.fromarray(data[i]).convert("RGB").save(savepath)
