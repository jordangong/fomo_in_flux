import json
import os
import tarfile

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/SynthCLIP106_classnames.json", "r"))
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
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "SynthCLIP106")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        assert_str = f"""
        There is currently no automatic dataset downloader for SynthCLIP. Please go to https://drive.google.com/file/d/1RP2nXOa0RDosUo_62b5TSZdGw8J13E4L/view,
        download manually and place the <SynthCLIP106.tar> inside {self.root.replace('/SynthCLIP106', '')}.
        """
        assert os.path.isdir(self.root), assert_str

        if os.path.isdir(self.root) and os.path.exists(f"{self.root}/images"):
            return

        assert_str = f"Can not find <SynthCLIP106.tar> in {self.root.replace('/SynthCLIP106', '')}"
        assert "SynthCLIP106.tar" in os.listdir(
            "/".join(self.root.split("/")[:-1])
        ), assert_str

        archive_file = os.path.join(
            "/".join(self.root.split("/")[:-1]), "SynthCLIP106.tar"
        )

        with tarfile.open(archive_file, "r") as tar:
            tar.extractall(path=self.root)

        assert os.path.isdir(
            self.root
        ), "SynthCLIP106 data is not setup correctly, please delete and try again."
        assert os.path.isdir(
            os.path.join(self.root, "images")
        ), "SynthCLIP106 data is not setup correctly, please delete and try again."
        assert (
            len(os.listdir(os.path.join(self.root, "images"))) == 106
        ), "SynthCLIP106 data is not setup correctly, please delete and try again."
