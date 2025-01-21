import json
import os
from typing import Tuple

import medmnist
import numpy as np
from PIL import Image
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/MedMNISTorganc_classnames.json", "r"))
PRIMER = "A photo of a {} CT scan."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4932, 0.4932, 0.4932),
    "std": (0.2839, 0.2839, 0.2839),
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

        self.root = os.path.join(root, "MedMNISTorganc")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        os.makedirs(self.root, exist_ok=True)
        image_folder = f"{self.root}/medmnistorganc-images-{self.split}"
        if not os.path.exists(image_folder):
            self.medmnist_organc_dataset = medmnist.OrganCMNIST(
                split=self.split, root=self.root, download=self.download
            )
            data = self.medmnist_organc_dataset.imgs
            data = np.expand_dims(data, axis=-1)
            data = np.concatenate([data for _ in range(3)], axis=-1)
            targets = self.medmnist_organc_dataset.labels.reshape(-1)

            os.makedirs(image_folder, exist_ok=True)
            for i in tqdm.trange(len(data), desc="Creating MedMNISTorganc images..."):
                savepath = f"{image_folder}/{targets[i]}-{i}.png"
                _ = Image.fromarray(data[i]).convert("RGB").save(savepath)
