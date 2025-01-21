import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/MVTECAD_Eval_classnames.json", "r"))
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

        self.root = os.path.join(root, "MVTECAD_Eval")
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

            # Create Adapt Folders
            if not os.path.exists(os.path.join(self.root, "..", "MVTECAD_Adapt")):
                os.makedirs(
                    os.path.join(self.root, "MVTECAD_Adapt", "data"), exist_ok=True
                )
                for folder in tqdm.tqdm(folders):
                    os.system(
                        f'chmod 777 {os.path.join(self.root, folder, "train", "good")}'
                    )
                    os.system(
                        f'chmod 777 {os.path.join(self.root, folder, "train", "good")}/*'
                    )
                    os.system(
                        f'cp -r {os.path.join(self.root, folder, "train", "good")} {os.path.join(self.root, "MVTECAD_Adapt", "data")}'
                    )
                    os.system(
                        f'mv {os.path.join(self.root, "MVTECAD_Adapt", "data", "good")} {os.path.join(self.root, "MVTECAD_Adapt", "data", folder)}'
                    )
                os.system(
                    f'mv {os.path.join(self.root, "MVTECAD_Adapt")} {os.path.join(self.root, "..")}'
                )

            # Create Eval Folders
            for folder in tqdm.tqdm(folders):
                subfolders = sorted(os.listdir(os.path.join(self.root, folder, "test")))
                for subfolder in subfolders:
                    os.system(
                        f'chmod 777 {os.path.join(self.root, folder, "test", subfolder)}'
                    )
                    os.system(
                        f'chmod 777 {os.path.join(self.root, folder, "test", subfolder)}/*'
                    )
                    os.system(
                        f'cp -r {os.path.join(self.root, folder, "test", subfolder)} {os.path.join(self.root, "data")}'
                    )
                    foldername = f"{folder}-{subfolder}"
                    os.system(
                        f'mv {os.path.join(self.root, "data", subfolder)} {os.path.join(self.root, "data", foldername)}'
                    )



backward_classname_compatibility = {
    'bottle-broken_large': 'bottle with large break',
    'bottle-broken_small': 'bottle with small break',
    'bottle-contamination': 'bottle with contamination',
    'bottle-good': 'bottle',
    'cable-bent_wire': 'cable with bent wire',
    'cable-cable_swap': 'cable with inner cable swap',
    'cable-combined': 'cable with combined',
    'cable-cut_inner_insulation': 'cable with cut inner insulation',
    'cable-cut_outer_insulation': 'cable with cut outer insulation',
    'cable-good': 'cable',
    'cable-missing_cable': 'cable with missing inner cable',
    'cable-missing_wire': 'cable with missing wire',
    'cable-poke_insulation': 'cable with poke insulation',
    'capsule-crack': 'capsule with crack',
    'capsule-faulty_imprint': 'capsule with faulty imprint',
    'capsule-good': 'capsule',
    'capsule-poke': 'capsule with poke',
    'capsule-scratch': 'capsule with scratched',
    'capsule-squeeze': 'capsule with squeeze',
    'carpet-color': 'carpet with colored spots',
    'carpet-cut': 'carpet with cut',
    'carpet-good': 'carpet',
    'carpet-hole': 'carpet with hole',
    'carpet-metal_contamination': 'carpet with metal contamination',
    'carpet-thread': 'carpet with thread',
    'grid-bent': 'grid with bent',
    'grid-broken': 'grid with broken',
    'grid-glue': 'grid with glue',
    'grid-good': 'grid',
    'grid-metal_contamination': 'grid with metal contamination',
    'grid-thread': 'grid with thread',
    'hazelnut-crack': 'hazelnut with crack',
    'hazelnut-cut': 'hazelnut with cut',
    'hazelnut-good': 'hazelnut',
    'hazelnut-hole': 'hazelnut with hole',
    'hazelnut-print': 'hazelnut with print',
    'leather-color': 'leather with colored spots',
    'leather-cut': 'leather with cut',
    'leather-fold': 'leather with fold',
    'leather-glue': 'leather with glue',
    'leather-good': 'leather',
    'leather-poke': 'leather with poke',
    'metal_nut-bent': 'metal nut with bent',
    'metal_nut-color': 'metal nut with colored spots',
    'metal_nut-flip': 'metal nut with flip',
    'metal_nut-good': 'metal nut',
    'metal_nut-scratch': 'metal nut with scratched',
    'pill-color': 'pill with colored spots',
    'pill-combined': 'pill with combined',
    'pill-contamination': 'pill with contamination',
    'pill-crack': 'pill with crack',
    'pill-faulty_imprint': 'pill with faulty imprint',
    'pill-good': 'pill',
    'pill-pill_type': 'pill with pill type',
    'pill-scratch': 'pill with scratched',
    'screw-good': 'screw',
    'screw-manipulated_front': 'screw with manipulated front',
    'screw-scratch_head': 'screw with scratched head',
    'screw-scratch_neck': 'screw with scratched neck',
    'screw-thread_side': 'screw with thread side',
    'screw-thread_top': 'screw with thread top',
    'tile-crack': 'tile with crack',
    'tile-glue_strip': 'tile with glue strip',
    'tile-good': 'tile',
    'tile-gray_stroke': 'tile with gray stroke',
    'tile-oil': 'tile with oil',
    'tile-rough': 'tile with rough',
    'toothbrush-defective': 'defective toothbrush',
    'toothbrush-good': 'toothbrush',
    'transistor-bent_lead': 'transistor with bent lead',
    'transistor-cut_lead': 'transistor with cut lead',
    'transistor-damaged_case': 'transistor with damaged case',
    'transistor-good': 'transistor',
    'transistor-misplaced': 'missing transistor',
    'wood-color': 'wood with colored spots',
    'wood-combined': 'wood with combined',
    'wood-good': 'wood',
    'wood-hole': 'wood with hole',
    'wood-liquid': 'wood with liquid',
    'wood-scratch': 'wood with scratched',
    'zipper-broken_teeth': 'zipper with broken teeth',
    'zipper-combined': 'zipper with combined',
    'zipper-fabric_border': 'zipper with fabric border',
    'zipper-fabric_interior': 'zipper with fabric interior',
    'zipper-good': 'zipper',
    'zipper-rough': 'zipper with rough',
    'zipper-split_teeth': 'zipper with split teeth',
    'zipper-squeezed_teeth': 'zipper with squeezed teeth'
}