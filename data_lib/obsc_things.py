import os

import json
import numpy as np
import pandas as pd

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/OBSC_THINGS_classnames.json", "r"))
PRIMER = "A photo of the obscure thing {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.3883, 0.3619, 0.3352),
    "std": (0.2378, 0.2189, 0.2055),
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

        self.root = os.path.join(root, "OBSC_THINGS")
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
