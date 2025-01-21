import os

import tqdm
from imagenetv2_pytorch import ImageNetV2Dataset

import data_lib
import data_lib.imagenet

BASE_CLASSES = sorted(data_lib.imagenet.BASE_CLASSES, key=lambda x: x.lower())
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

        assert not self.train, "No training data available for Imagenet-V2!"

        self.root = os.path.join(root, "IMAGENET_V2")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        os.makedirs(self.root, exist_ok=True)
        self.imagenetv2_dataset = ImageNetV2Dataset(location=self.root)

        if not os.path.exists(os.path.join(self.root, self.split)):
            data = [str(x) for x in self.imagenetv2_dataset.fnames]
            targets = [
                int(str(x).split("/")[-2]) for x in self.imagenetv2_dataset.fnames
            ]

            os.makedirs(os.path.join(self.root, self.split), exist_ok=True)
            classes = data_lib.imagenet.IMAGENET_CLASSES
            for filepath, target in tqdm.tqdm(
                zip(data, targets), total=len(data), desc="Moving data..."
            ):
                classname = classes[target].replace("/", "-")
                class_folder = os.path.join(self.root, self.split, classname)
                os.makedirs(class_folder, exist_ok=True)
                os.system(f'cp {filepath} "{class_folder}"')

        data = []
        targets = []
        classes = sorted(
            os.listdir(os.path.join(self.root, self.split)), key=lambda x: x.lower()
        )
        for i, folder in enumerate(classes):
            for file in os.listdir(os.path.join(self.root, self.split, folder)):
                data.append(os.path.join(self.root, self.split, folder, file))
                targets.append(i)

        self.data = data
        self.targets = targets
