import json
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/SVHN_classnames.json", "r"))
PRIMER = 'A photo of the number: "{}".'
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4377, 0.4438, 0.4728),
    "std": (0.1980, 0.2010, 0.1970),
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

        self.root = os.path.join(root, "SVHN")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        image_folder = f"{self.root}/svhn-images-{self.split}"
        if not os.path.exists(image_folder):
            self.svhn_dataset = torchvision.datasets.SVHN(
                self.root,
                self.split,
                transform=None,
                target_transform=None,
                download=self.download,
            )

            data = np.array(self.svhn_dataset.data.transpose(0, 2, 3, 1))
            targets = np.array(self.svhn_dataset.labels)
            os.makedirs(image_folder, exist_ok=True)
            for i in tqdm.trange(len(data), desc="Creating SVHN images..."):
                savepath = f"{image_folder}/{targets[i]}-{i}.png"
                _ = Image.fromarray(data[i]).convert("RGB").save(savepath)
