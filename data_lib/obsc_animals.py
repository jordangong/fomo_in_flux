import os

import json
import numpy as np
import pandas as pd

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/OBSC_ANIMALS_classnames.json", "r"))
PRIMER = "A photo of the obscure animal {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.7838, 0.7188, 0.6859),
    "std": (0.2566, 0.2637, 0.2836),
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

        self.root = os.path.join(root, "OBSC_ANIMALS")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            raise NotImplementedError(
                "Currently no auto-setup option for dataset [obsc_animals]."
            )
