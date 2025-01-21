import json
import os
from typing import Tuple

import numpy as np
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/STL10_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4467, 0.4398, 0.4066),
    "std": (0.2603, 0.2566, 0.2713),
    # Default Imagesize
    "img_size": 96,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "STL10")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        image_folder = f"{self.root}/stl10-images-{self.split}"
        if not os.path.exists(image_folder):
            self.stl10_dataset = torchvision.datasets.STL10(
                self.root,
                self.split,
                transform=None,
                target_transform=None,
                download=self.download,
            )

            data = np.array(self.stl10_dataset.data.transpose(0, 2, 3, 1))
            targets = np.array(self.stl10_dataset.labels)
            os.makedirs(image_folder, exist_ok=True)
            for i in tqdm.trange(len(data), desc="Creating STL10 images..."):
                savepath = f"{image_folder}/{targets[i]}-{i}.png"
                _ = Image.fromarray(data[i]).convert("RGB").save(savepath)
