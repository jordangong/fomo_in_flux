import json
import os

import gdown

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/IMAGENET_S_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.8533, 0.8531, 0.8539),
    "std": (0.2175, 0.2178, 0.2172),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": True,
    "primer": PRIMER,
    "eval_only": True,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        assert not self.train, "No training data available for Imagenet-S!"

        self.root = os.path.join(root, "IMAGENET_S")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            os.makedirs(self.root, exist_ok=True)
            gdown.download(
                "https://drive.google.com/u/0/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA",
                os.path.join(self.root, "data.zip"),
                quiet=False,
            )
            os.system(f"unzip -q {self.root}/data.zip")
