import json
import os

import torchvision.transforms

import data_lib

BASE_CLASSES = json.load(
    open("data_lib/00_info/DOMAINNET_PAINTING_classnames.json", "r")
)
PRIMER = "A painting of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5734, 0.5455, 0.5067),
    "std": (0.2255, 0.2190, 0.2243),
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

        self.root = os.path.join(root, "DOMAINNET_PAINTING")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("Downloading dataset...")
            dataset_url = (
                "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip"
            )
            train_split = "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt"
            test_split = "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt"
            torchvision.datasets.utils.download_and_extract_archive(
                dataset_url, download_root=self.root
            )
            os.system(f"wget -O {self.root}/painting_train.txt {train_split}")
            os.system(f"wget -O {self.root}/painting_test.txt {test_split}")
