import json
import os

import pandas as pd
import zipfile

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/MONKEYS10_classnames.json", "r"))
PRIMER = "A photo of a {}, a type of monkey."
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
    "eval_only": False,
    "type": "classification",
}


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "MONKEYS10")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):

        os.makedirs(self.root, exist_ok=True)

        if not os.path.exists(os.path.join(self.root, "monkeys10")):
            assert_str = f"""
            There is currently no automatic dataset downloader for MONKEYS10. Please download from https://www.kaggle.com/datasets/slothkong/10-monkey-species/download?datasetVersionNumber=2, 
            place the <archive.zip> inside {self.root}.
            """
            if (
                not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
                and self.download
            ):
                raise NotImplementedError(assert_str)

            os.makedirs(os.path.join(self.root, "monkeys10"), exist_ok=True)

            req_archive_setup = True
            if (
                "training" in os.listdir(os.path.join(self.root, "monkeys10"))
                and "validation" in os.listdir(os.path.join(self.root, "monkeys10"))
                and "monkey_labels.txt"
                in os.listdir(os.path.join(self.root, "monkeys10"))
            ):
                req_archive_setup = False

            if req_archive_setup:
                assert "archive.zip" in os.listdir(
                    self.root
                ), "<archive.zip> does not exist in {}, please download it from https://www.kaggle.com/datasets/slothkong/10-monkey-species and place it here: {}".format(
                    self.root, self.root
                )

                archive_file = os.path.join(self.root, "archive.zip")

                with zipfile.ZipFile(archive_file, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(self.root, "monkeys10"))

            assert "training" in os.listdir(
                os.path.join(self.root, "monkeys10")
            ) and "validation" in os.listdir(
                os.path.join(self.root, "monkeys10")
            ), "Dataset files are corrupted or not correctly extracted, please download and place <archive.zip> from https://www.kaggle.com/datasets/slothkong/10-monkey-species again inside {}".format(
                self.root
            )

            label_df = pd.read_csv(
                os.path.join(self.root, "monkeys10", "monkey_labels.txt"),
                skipinitialspace=True,
            ).to_numpy()
            i2c_mapping = {
                int(l[0].strip()[-1]): l[2].strip().replace("_", " ") for l in label_df
            }
            c2i_mapping = {v: k for k, v in i2c_mapping.items()}

            self.data = []
            self.targets = []

            target_folder = (
                os.path.join(self.root, "monkeys10", "training", "training")
                if self.train
                else os.path.join(self.root, "monkeys10", "validation", "validation")
            )

            for index, folder in enumerate(os.listdir(target_folder)):
                for file in os.listdir(os.path.join(target_folder, folder)):
                    # remove all corrupted files / non image files
                    if ".jpg" in file or ".png" in file:
                        self.data.append(os.path.join(target_folder, folder, file))
                        self.targets.append(index)
