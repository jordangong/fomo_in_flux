import json
import os
from typing import Tuple

from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/CIFAR10_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2470, 0.2435, 0.2615),
    # Default Imagesize
    "img_size": 32,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "CIFAR10")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        img_folder = f"{self.root}/cifar-10-images-{self.split}"
        if not os.path.exists(img_folder):
            cifar10_dataset = torchvision.datasets.CIFAR10(
                self.root,
                self.train,
                transform=None,
                target_transform=None,
                download=self.download,
            )
            os.makedirs(img_folder, exist_ok=True)
            for i in tqdm.trange(
                len(cifar10_dataset.data), desc="Converting CIFAR10 data to images..."
            ):
                savepath = f"{img_folder}/{i}.png"
                Image.fromarray(cifar10_dataset.data[i]).convert("RGB").save(savepath)
