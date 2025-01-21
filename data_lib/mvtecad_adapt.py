import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/MVTECAD_Adapt_classnames.json", "r"))
PRIMER = "A close-up photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.3594, 0.3516, 0.3517),
    "std": (0.1844, 0.1809, 0.1642),
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

        self.root = os.path.join(root, "MVTECAD_Adapt")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        if (
            not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
            and self.download
        ):
            print("[INFO] This will set up both MVTECAD_Adapt and MVTECAD_Eval!")
            print("Downloading dataset...")
            dataset_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
            torchvision.datasets.utils.download_and_extract_archive(
                dataset_url, download_root=self.root
            )

            folders = sorted(
                [
                    x
                    for x in os.listdir(self.root)
                    if "MVTECAD" not in x
                    and "." not in x
                    and x != "data"
                    and os.path.isdir(os.path.join(self.root, x))
                ]
            )

            os.makedirs(os.path.join(self.root, "data"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "MVTECAD_Eval", "data"), exist_ok=True)

            # Create Adapt Folders
            for folder in tqdm.tqdm(folders):
                os.system(
                    f'chmod 777 {os.path.join(self.root, folder, "train", "good")}'
                )
                os.system(
                    f'chmod 777 {os.path.join(self.root, folder, "train", "good")}/*'
                )
                os.system(
                    f'cp -r {os.path.join(self.root, folder, "train", "good")} {os.path.join(self.root, "data")}'
                )
                os.system(
                    f'mv {os.path.join(self.root, "data", "good")} {os.path.join(self.root, "data", folder)}'
                )

            # Create Eval Folders
            if not os.path.exists(os.path.join(self.root, "..", "MVTECAD_Eval")):
                for folder in tqdm.tqdm(folders):
                    subfolders = sorted(
                        os.listdir(os.path.join(self.root, folder, "test"))
                    )
                    for subfolder in subfolders:
                        os.system(
                            f'chmod 777 {os.path.join(self.root, folder, "test", subfolder)}'
                        )
                        os.system(
                            f'chmod 777 {os.path.join(self.root, folder, "test", subfolder)}/*'
                        )
                        os.system(
                            f'cp -r {os.path.join(self.root, folder, "test", subfolder)} {os.path.join(self.root, "MVTECAD_Eval", "data")}'
                        )
                        foldername = f"{folder}-{subfolder}"
                        os.system(
                            f'mv {os.path.join(self.root, "MVTECAD_Eval", "data", subfolder)} {os.path.join(self.root, "MVTECAD_Eval", "data", foldername)}'
                        )
                os.system(
                    f'mv {os.path.join(self.root, "MVTECAD_Eval")} {os.path.join(self.root, "..")}'
                )
