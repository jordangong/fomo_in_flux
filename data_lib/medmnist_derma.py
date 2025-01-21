import json
import os
from typing import Tuple

import medmnist
import numpy as np
from PIL import Image
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/MedMNISTderma_classnames.json", "r"))
PRIMER = "A dermatologic photo of {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.7631, 0.5381, 0.5614),
    "std": (0.1366, 0.1543, 0.1692),
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

        self.root = os.path.join(root, "MedMNISTderma")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        os.makedirs(self.root, exist_ok=True)
        image_folder = f"{self.root}/medmnistderma-images-{self.split}"
        if not os.path.exists(image_folder):
            self.medmnist_derma_dataset = medmnist.DermaMNIST(
                split=self.split, root=self.root, download=self.download
            )
            data = np.array(self.medmnist_derma_dataset.imgs)
            targets = np.array(self.medmnist_derma_dataset.labels.reshape(-1))

            os.makedirs(image_folder, exist_ok=True)
            for i in tqdm.trange(len(data), desc="Creating MedMNISTderma images..."):
                savepath = f"{image_folder}/{targets[i]}-{i}.png"
                _ = Image.fromarray(data[i]).convert("RGB").save(savepath)
