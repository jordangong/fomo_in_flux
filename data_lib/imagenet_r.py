import json
import os

import torchvision.datasets
import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/IMAGENET_R_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
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

        assert not self.train, "No training data available for Imagenet-R!"

        self.root = os.path.join(root, "IMAGENET_R")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading dataset...")
            dataset_url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
            torchvision.datasets.utils.download_and_extract_archive(
                dataset_url, download_root=self.root
            )
