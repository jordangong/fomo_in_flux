import json
import os

import h5py
import numpy as np
from PIL import Image
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/SHAPES3D_classnames.json", "r"))
PRIMER = "A photo of a synthetic {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5066, 0.5805, 0.6005),
    "std": (0.2921, 0.3447, 0.3696),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}

lat_names = ("floorCol", "wallCol", "objCol", "objSize", "objType", "objAzimuth")
lat_sizes = np.array([10, 10, 10, 8, 4, 15])


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "SHAPES3D")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        image_folder = f"{self.root}/shapes3d-images-{self.split}"
        if not os.path.exists(image_folder):
            if (
                not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
                and self.download
            ):
                os.makedirs(self.root, exist_ok=True)
                base_url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
                os.system(f"wget -O {self.root}/shapes3d.h5 {base_url}")

            with h5py.File(f"{self.root}/shapes3d.h5", "r") as shapes3d_data:
                imgs = shapes3d_data["images"][()]
                lat_values = shapes3d_data["labels"][()]

            files = json.load(open(f"data_lib/00_info/SHAPES3D_{self.split}.json", "r"))
            files = list(files.keys())
            indices = [int(x.split("/")[-1].split(".")[0]) for x in files]

            os.makedirs(image_folder, exist_ok=True)
            for i in tqdm.tqdm(indices, desc="Creating Shapes3D images..."):
                savepath = f"{image_folder}/{i}.png"
                _ = Image.fromarray(imgs[i]).convert("RGB").save(savepath)
