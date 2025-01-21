import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import shutil
import termcolor
import zipfile

import data_lib

BASE_CLASSES = json.load(open("data_lib/00_info/VEG200_classnames.json", "r"))
# BASE_CLASSES = [
#     'achyranthes', 'adenophora', 'agaricus bisporus', 'agaricus blazei murill', 'agrimony', 'agrocybe aegerita', 'allium', 'arrowhead', 'artemisia selengensis', 'asparagus', 'asparagus fern', 'asparagus lettuce', 'asparagus pea', 'azuki beans', 'balsam pear',
#     'bamboo shoot', 'basella rubra', 'basil', 'bassia scoparia', 'beefsteak plant', 'beetroot', 'bird pepper', 'black bean sprouts', 'black salsify', 'black soya bean', 'bolete', 'bottle gourd', 'broad bean', 'broccoli', 'brussels sprouts', 'bunching onion',
#     'burclover', 'burdock root', 'cape gooseberry', 'carduus', 'carrot', 'cattail', 'celeriac', 'celery', 'centella asiatica', 'chantarelle', 'chicory', 'Chinese artichoke', 'Chinese cabbage', 'Chinese kale', 'Chinese mallow', 'Chinese pumpkin', 'Chinese yam',
#     'chive', 'chocho', 'chrysanthemum', 'commelina', 'coprinus comatus', 'coriander', 'corn', 'cowpea', 'cress', 'cucumber', 'cudweed', 'curly kale', 'cynoglossum lanceolatum', 'dandelion', 'day lily', 'dictyophora', 'edible amaranth', 'eggplant', 'endive',
#     'enoki mushroom', 'equisetum debile', 'fallopia multiflora', 'feather cockscomb', 'fennel', 'flower Chinese cabbage', 'galinsoga parviflora', 'garlic', 'garlic chive', 'garlic sprouts', 'ginger', 'globe artichoke', 'goji berry', 'gorgon fruit seed', 'gourd',
#     "great Solomon's-seal", 'green Chinese onion', 'green eggplant', 'green radish', 'gynura bicolor', 'hairy squash', 'head cabbage', 'hen-of-the-woods', 'Herb of Ghostplant Wormwood', 'hericium', 'horst', 'houttuynia cordata', 'hyacinth bean', 'hypsizigus marmoreus',
#     'jerusalem artichoke', "Jew's-ear", 'kalimeris', 'kidney bean', 'kidney bean seed', 'kohlrabi', 'konnyaku', 'kudzu', 'leaf lettuce', 'leek', 'lettuce', 'Lily', 'lotus', 'lotus root', 'lotus seed', 'lotus seedpod', 'luffa acutangula', 'luffa cylindrica', 'matsutake',
#     'milk thistle', 'mint', 'mioga ginger', 'mitsuba', 'morel', 'mung bean', 'mung bean sprouts', 'mustard', 'nameko', 'nankimgense', 'New Zealand spinach', 'okra', 'onion', 'ostrich fern', 'oyster mushroom', 'pakchoi', 'parsley', 'parsnip', 'pea', 'peanut sprouts',
#     'pepper', 'pimento', 'platycodon grandiflorum', 'pleurotus eryngii', 'pleurotus nebrodensis', 'polygonatum sibiricum', 'polygonum lapathifolium', 'potato', 'prickly lettuce', 'pumpkin', 'purple cai-tai', 'purslane', 'red cabbage', 'red radish', 'rhubarb',
#     'russula virescens', 'savoy caggage', 'scallion', 'sea of nostoc flagelliforme', 'self-heal', 'shallot', "shepherd's purse", 'shiitake', 'sieva bean', 'sieva bean seed', 'silverweed', 'snake gourd', 'sorrel', 'soybean', 'soybean seed', 'soybean sprouts',
#     'spinach', 'sprouting broccoli', 'straw mushroom', 'strawberry', 'sweet potato', 'swiss chard', 'sword bean', 'taro', 'termite mushroom', 'thorny amaranth', 'tomato', 'toon', 'tremella fuciformis', 'tricholoma flavovirens', 'turnip cabbage', 'vetch',
#     'viola philippica', 'wasabi', 'water caltrop', 'water chestnuts', 'water shield', 'water spinach', 'watercress', 'watermelon', 'wax gourd', 'white eggplant', 'white radish', 'wild amaranth', 'wild chrysanthemum', 'Wuta-tsai', 'yam bean', 'zha-tsai',
#     'zizania aquatica', 'zucchini'
# ]
PRIMER = "A photo of a {}, a type of vegetable."
CLASSES = [PRIMER.format(x) for x in BASE_CLASSES]

PARAMS = {
    "base_classes": BASE_CLASSES,
    "classes": CLASSES,
    "num_classes": len(CLASSES),
    "mean": (0.5220, 0.5319, 0.3973),
    "std": (0.2151, 0.2172, 0.2267),
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

        self.root = os.path.join(root, "VEG200")
        self.PARAMS = PARAMS

        self._run_default_setup()

    def _download_and_setup(self):
        fru92_root = self.root.replace("VEG200", "FRU92")

        assert_str = f"""
        There is currently no automatic dataset downloader for VEG200/FRU92. Please go to https://www.kaggle.com/datasets/zhaoyj688/vegfru,
        download manually and place the <archive.zip> inside either {self.root} or {fru92_root}.
        """
        assert os.path.isdir(self.root), assert_str
        os.makedirs(fru92_root, exist_ok=True)
        os.makedirs(os.path.join(fru92_root, "processed"), exist_ok=True)

        # Setting up datasets across FRU92 and VEG200.
        basefiles_path = os.path.join(self.root, "processed", "vegfru_list")
        if "processed" not in os.listdir(self.root):
            termcolor.cprint(
                "Note: Setting up VEG200 automatically sets up FRU92.",
                "yellow",
                attrs=["bold"],
            )
            assert_str = (
                f"Can not find <archive.zip> in either {self.root} or {fru92_root}"
            )
            assert "archive.zip" in os.listdir(
                self.root
            ) or "archive.zip" in os.listdir(fru92_root), assert_str

            if "archive.zip" in os.listdir(self.root):
                archive_file = os.path.join(self.root, "archive.zip")
            else:
                archive_file = os.path.join(fru92_root, "archive.zip")

            with zipfile.ZipFile(archive_file, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.root, "processed"))

            veg200_fru_path = os.path.join(self.root, "processed", "fru92_images")
            shutil.move(veg200_fru_path, os.path.join(fru92_root, "processed"))
            shutil.copytree(
                basefiles_path, os.path.join(fru92_root, "processed", "vegfru_list")
            )
