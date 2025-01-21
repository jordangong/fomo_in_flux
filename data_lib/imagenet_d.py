import json
import os
import tarfile

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/IMAGENET_D_classnames.json", "r"))
PRIMER = "A photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    ### Note: using default ImageNet mean and std -- does not matter for continual pretraining as overwritten by pretrained model mean and stds
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

        self.root = os.path.join(root, "IMAGENET_D")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        assert_str = f"""
        There is currently no automatic dataset downloader for IMAGENET_D. Please go to https://drive.google.com/file/d/1x4U0BgTFBaHQ_xNPedhbCAo3R81B7AAm/view,
        download manually and place the <ImageNet-D.tar> inside {self.root}.
        """
        assert os.path.isdir(self.root), assert_str

        if os.path.isdir(os.path.join(self.root, "ImageNet-D")):
            return

        assert_str = f"Can not find <ImageNet-D.tar> in {self.root}"
        assert "ImageNet-D.tar" in os.listdir(self.root), assert_str

        archive_file = os.path.join(self.root, "ImageNet-D.tar")

        with tarfile.open(archive_file, "r") as tar:
            tar.extractall(path=self.root)

        assert os.path.isdir(
            self.root
        ), "IMAGENET_D data is not setup correctly, please delete and try again."
        assert os.path.isdir(
            os.path.join(self.root, "ImageNet-D")
        ), "IMAGENET_D data is not setup correctly, please delete and try again."
        assert os.path.isdir(
            os.path.join(self.root, "ImageNet-D", "merged")
        ), "IMAGENET_D data is not setup correctly, please delete and try again."
        assert (
            len(os.listdir(os.path.join(self.root, "ImageNet-D", "merged"))) == 103
        ), "IMAGENET_D data is not setup correctly, please delete and try again."
