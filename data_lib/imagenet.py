import json
import os

import data_lib

BASE_CLASSES = IMAGENET_CLASSES = json.load(
    open("data_lib/00_info/IMAGENET_classnames.json", "r")
)
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

        assert not self.train, "No training data available for Imagenet!"

        self.root = os.path.join(self.root, "IMAGENET")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        pass
        # self.imagenet_dataset = torchvision.datasets.ImageNet(self.root, split=self.split)
        # self.data = [x[0] for x in self.imagenet_dataset.imgs]
        # self.targets = [x[1] for x in self.imagenet_dataset.imgs]
        # self.data = []
        # self.targets = []
        # for i, folder in enumerate(sorted(os.listdir(os.path.join(self.root, self.split)))):
        #     if folder != 'README.txt':
        #         for file in os.listdir(os.path.join(self.root, self.split, folder)):
        #             self.data.append(os.path.join(self.root, self.split, folder, file))
        #             self.targets.append(i)
