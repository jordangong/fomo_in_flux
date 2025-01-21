import json
import os
from typing import Tuple

import itertools as it
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/DSPRITES_classnames.json", "r"))

PRIMER = "A black-and-white photo of a {}."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.043, 0.043, 0.043),
    "std": (0.202, 0.202, 0.202),
    # Default Imagesize
    "img_size": 224,
    # If it's computationally recommended to resize in advance
    "create_resized_variant_if_possible": False,
    "primer": PRIMER,
    "eval_only": False,
    "type": "classification",
}

lat_names = ("color", "shape", "scale", "orientation", "posX", "posY")
lat_sizes = np.array([1, 3, 6, 40, 32, 32])


class Dataset(data_lib.DatasetScaffold):
    def __init__(self, root, **kwargs) -> None:
        super(Dataset, self).__init__(root, **kwargs)

        self.root = os.path.join(root, "DSPRITES")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        image_folder = f"{self.root}/dsprites-images-{self.split}"
        if not os.path.exists(image_folder):

            if (
                not (os.path.isdir(self.root) and len(os.listdir(self.root)) > 0)
                and self.download
            ):
                os.makedirs(self.root, exist_ok=True)
                base_url = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
                os.system(f"wget -O {self.root}/dsprites.npz {base_url}")
            compressed_data = np.load(f"{self.root}/dsprites.npz")
            imgs = compressed_data["imgs"]
            latent_values = compressed_data["latents_values"]

            # Only use top-9, center-9 and bottom-9 x/y coordinates.
            # 0-9, 12-21, 23-32
            rem = np.zeros(len(latent_values))
            for k in range(1, 3):
                vals = np.unique(latent_values[:, -k])
                for i in [9, 10, 11, 21, 22]:
                    rem[np.where(latent_values[:, -k] == vals[i])[0]] = 1
            retain = (1 - rem).astype(bool)
            sub_imgs = imgs[retain]
            sub_latents = latent_values[retain]

            np.random.seed(0)
            avail_latents = np.array(
                list(
                    it.product(
                        range(1), range(3), range(6), range(40), range(27), range(27)
                    )
                )
            )
            train_indices = list(
                np.random.choice(len(avail_latents), 75000, replace=False)
            )
            test_indices = list(
                np.random.choice(
                    list(set(range(len(avail_latents))) - set(train_indices)),
                    25000,
                    replace=False,
                )
            )
            train_latents = avail_latents[sorted(train_indices)]
            test_latents = avail_latents[sorted(test_indices)]
            mul = np.array([[524880, 174960, 29160, 729, 27, 1]])
            train_indices = np.sum(train_latents * mul, axis=-1)
            test_indices = np.sum(test_latents * mul, axis=-1)

            # bottom, center, top
            # left, center, right
            ll = []
            for lab in tqdm.tqdm(sub_latents, desc="Generating label data..."):
                x = int(lab[-2] > 0.33) + int(lab[-2] > 0.66)
                y = int(lab[-1] > 0.33) + int(lab[-1] > 0.66)
                s = int(lab[1])
                label = (s - 1) * 9 + x * 3 + y
                ll.append(label)
            ll = np.array(ll).astype(int)

            files = json.load(open(f"data_lib/00_info/DSPRITES_{self.split}.json", "r"))
            files = list(files.keys())
            indices = [int(x.split("/")[-1].split(".")[0]) for x in files]

            os.makedirs(image_folder, exist_ok=True)
            for i in tqdm.tqdm(indices, desc="Creating DSPRITES images..."):
                savepath = f"{image_folder}/{i}.png"
                _ = Image.fromarray(sub_imgs[i] * 255).convert("RGB").save(savepath)

        # self.data = np.expand_dims(np.load(f'{self.root}/dsprites_data_{self.split}.npy'), axis=-1)
        # self.data = np.concatenate([self.data for _  in range(3)], axis=-1)
        # self.targets = torch.from_numpy(np.load(f'{self.root}/dsprites_labels_{self.split}.npy')).to(torch.long)

        # Obj 0-2: [Square, Ellipse, Heart]
        # Top Left Square, Center left Square, Bottom Left Square, Top Center Square, Central Square, Bottom Center Square, Top Right Square, Center Right Square, Bottom Right Square,
        # Top Left Square, Center left Ellipse, Bottom Left Ellipse, Top Center Ellipse, Central Ellipse, Bottom Center Ellipse, Top Right Ellipse, Center Right Ellipse, Bottom Right Ellipse,
        # Top Left Heart, Center left Heart, Bottom Left Heart, Top Center Heart, Central Heart, Bottom Center Heart, Top Right Heart, Center Right Heart, Bottom Right Heart
