import json
import os
from typing import Tuple

import json
import numpy as np
from PIL import Image
import termcolor
import tqdm
import zipfile

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/MTSD_classnames.json", "r"))
PRIMER = 'A centered photo of a "{}" traffic sign.'
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.4118, 0.3928, 0.3629),
    "std": (0.1696, 0.1664, 0.1550),
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

        self.root = os.path.join(root, "MTSD")
        self.PARAMS = PARAMS
        self.split = "train" if self.train else "val"

        self._run_default_setup()

    def _download_and_setup(self):
        assert_str = f"""
        MTSD has to be downloaded manually from https://www.mapillary.com/dataset/trafficsign, particularly the six zip-files:
        * mtsd_fully_annotated_annotation.zip
        * mtsd_fully_annotated_images.test.zip
        * mtsd_fully_annotated_images.train.0.zip
        * mtsd_fully_annotated_images.train.1.zip
        * mtsd_fully_annotated_images.train.2.zip
        * mtsd_fully_annotated_images.val.zip
        After downloading, simply place them in {self.root}, the dataloader will handle the rest.
        """
        assert os.path.isdir(self.root), assert_str

        if "MTSD" not in os.listdir(self.root):
            os.makedirs(os.path.join(self.root, "MTSD"), exist_ok=True)
            for zipf in tqdm.tqdm(os.listdir(self.root), desc="Unzipping files..."):
                if zipf != "MTSD":
                    with zipfile.ZipFile(os.path.join(self.root, zipf), "r") as zip_ref:
                        zip_ref.extractall(os.path.join(self.root, "MTSD"))

        if "processed" not in os.listdir(
            os.path.join(self.root, "MTSD")
        ) or self.split not in os.listdir(os.path.join(self.root, "MTSD", "processed")):
            annot_dir = os.path.join(
                self.root, "MTSD", "mtsd_v2_fully_annotated", "annotations"
            )
            splits_dir = os.path.join(
                self.root, "MTSD", "mtsd_v2_fully_annotated", "splits"
            )

            with open(os.path.join(splits_dir, "{}.txt".format(self.split)), "r") as f:
                image_ids = [s.strip() for s in f.readlines()]

            termcolor.cprint(
                f"Preprocessing MTSD [{self.split}] - this can take a while...",
                "yellow",
                attrs=["bold"],
            )
            for image_id in tqdm.tqdm(image_ids, "Processing images..."):

                img_id = image_id

                annots_obj = json.load(
                    open(os.path.join(annot_dir, "{}.json".format(image_id)))
                )

                if annots_obj["ispano"]:
                    continue

                for o in annots_obj["objects"]:
                    label = o["label"]
                    if label == "other-sign":
                        continue

                    if "--" in label:
                        label = label.split("--")[1]

                    dirname = os.path.join(
                        self.root, "MTSD", "processed", self.split, label
                    )
                    os.makedirs(dirname, exist_ok=True)

                    bb = o["bbox"]
                    bb = [
                        round(bb["xmin"]),
                        round(bb["ymin"]),
                        round(bb["xmax"]),
                        round(bb["ymax"]),
                    ]

                    im = Image.open(
                        os.path.join(self.root, "MTSD", "images", img_id + ".jpg")
                    )
                    im = im.crop(bb)

                    im.save(os.path.join(dirname, img_id + "__" + o["key"] + ".jpg"))
