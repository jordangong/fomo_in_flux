import json
import os

import torchvision

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/CLEVR_classnames.json", "r"))
PRIMER = "A photo of synthetic shapes, with {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5066, 0.5805, 0.6005),
    "std": (0.2921, 0.3447, 0.3696),
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

        self.root = os.path.join(root, "CLEVR")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            data_url = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
            torchvision.datasets.utils.download_and_extract_archive(
                data_url, download_root=self.root
            )
