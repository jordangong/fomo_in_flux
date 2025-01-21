import json
import os

import torchvision.datasets
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/CALTECH101_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5454, 0.5259, 0.4989),
    "std": (0.2456, 0.2428, 0.2452),
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

        self.root = os.path.join(root, "CALTECH101")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        self.caltech101_dataset = torchvision.datasets.Caltech101(
            self.root, transform=None, target_transform=None, download=self.download
        )
