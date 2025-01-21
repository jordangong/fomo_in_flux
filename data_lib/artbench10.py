import json
import os

import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/ARTBENCH10_classnames.json", "r"))
PRIMER = "A painting of the artist {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0, 0, 0),
    "std": (1, 1, 1),
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

        self.root = os.path.join(root, "ARTBENCH10")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            data_url = "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar"
            torchvision.datasets.utils.download_and_extract_archive(
                data_url, download_root=self.root
            )
            metadata_url = "https://artbench.eecs.berkeley.edu/files/ArtBench-10.csv"
            os.system(f"wget -O {self.root}/metadata.csv {metadata_url}")
