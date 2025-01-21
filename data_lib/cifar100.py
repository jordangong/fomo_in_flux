import json
import os
from typing import Tuple

from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/CIFAR100_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5071, 0.4867, 0.4408),
    "std": (0.2675, 0.2565, 0.2761),
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

        self.root = os.path.join(root, "CIFAR100")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        img_folder = f"{self.root}/cifar-100-images-{self.split}"
        if not os.path.exists(img_folder):
            cifar100_dataset = torchvision.datasets.CIFAR100(
                self.root,
                self.train,
                transform=None,
                target_transform=None,
                download=self.download,
            )
            os.makedirs(img_folder, exist_ok=True)

            for i in tqdm.trange(
                len(cifar100_dataset.data), desc="Converting CIFAR100 data to images..."
            ):
                savepath = f"{img_folder}/{i}.png"
                Image.fromarray(cifar100_dataset.data[i]).convert("RGB").save(savepath)
