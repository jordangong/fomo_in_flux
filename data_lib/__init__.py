import importlib
import io
import json
import os
import tarfile
import time
from typing import Dict, Union, List

import numpy as np
import omegaconf
import pathlib
from PIL import Image, ImageFile
from omegaconf import open_dict
import termcolor
import torch
import torchvision
import tqdm
from torch.utils.data import Dataset
import random
try:
    from wds_utils import get_wds_dataset
except ImportError as e:
    from .wds_utils import get_wds_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
# to fix warnings like this:
# Image size (XXXXXXX pixels) exceeds limit of YYYYYYYY pixels
# , could be decompression bomb DOS attack.
Image.MAX_IMAGE_PIXELS = None

DATACOMP_SIZE_ = 1251473
# taken from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/constants.py
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def get_all_datasets():
    return sorted(
        [
            dataset.split(".")[0]
            for dataset in os.listdir("data_lib")
            if not dataset.find("__") > -1
            and "py" in dataset
            and "merge_utils" not in dataset
            and "miniimagenet" not in dataset
            and "wds_utils" not in dataset
        ]
    )

DATASETS = {dataset: f"data_lib.{dataset}" for dataset in get_all_datasets()}

class DatasetScaffold(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        args: omegaconf.DictConfig = None,
        train: bool = True,
        download: bool = True,
        transform: List[str] = ["ToTensor", "Normalize"],
        preload: bool = False,
        create_resized_data: bool = False,
        train_val_split: float = 0.8,
        captions_root: str = "data/dataset_captions",
        **kwargs,
    ) -> None:
        self.root = root
        self.args = args
        self.train = train
        self.transform_list = transform if transform is not None else []
        self.split = "train" if train else "test"

        # For Dataloader-initialization.
        self.download = download
        self.train_val_split = train_val_split
        self.captions_root = captions_root

        self.preload = preload
        self.preloaded_image_data = []
        self.create_resized_data = create_resized_data

        self.req_aux_inputs = False
        self.req_non_aug_inputs = False
        self.to_tensor = torchvision.transforms.ToTensor()

        self.PARAMS = {
            "mean": (0, 0, 0),
            "std": (1, 1, 1),
            "resize": -1,  # resize = -1 default to img_size for resizing.
            "img_size": 224,
            "classes": [],
            "num_classes": 100,
            "primer": "A photo of a {}.",
            "create_resized_variant_if_possible": False,
        }

        self.set_params()
        self.set_transforms()

        # Training/Test data & targets need to be exposed to
        # allow for conversion to sequential setup.
        # NOTE: THESE NEED TO BE UPDATED FOR EACH DATASET.
        self.info_dict = None  # Base data dict containing all relevant information.
        self.data = None  # List of image paths.
        self.targets = None  # Correspondingly assigned target values.
        # (optional) Caption Data
        self.caption_data = None  # Corresponding caption list.

        # Value to increase output target values by if multiple datasets are stacked.
        self.target_offset = 0

        # Flag. Set if resizing data is available and can be used directly.
        self.data_setup_required = True
        self.run_with_resized_data = False

    def _run_default_setup(self):
        # Populate general parameters and utilized transformations.
        self.set_params()
        self.set_transforms()

        # Load from already preprocessed data if available & desired.
        # If not, will download and set up data as needed.
        if not self.args.experiment.dataset.tar:
            self._check_for_preprocessed_data()
            if self.data_setup_required:
                self._download_and_setup()

        # Load dataset information json file, and populate relevant variables and attributes.
        self.info_dict = json.load(
            open(f'data_lib/00_info/{self.root.split("/")[-1]}_{self.split}.json', "r")
        )
        self.set_data_and_targets()

        # If resized data is available or resizing is needed:
        self.run_with_resized_data = (
            self.create_resized_data
            and isinstance(self.data[0], str)
            and self.PARAMS["create_resized_variant_if_possible"]
        )
        if self.run_with_resized_data:
            resized_folder = (
                f"{self.root.split('/')[-1]}_{self.split}_{self.PARAMS['resize']}"
            )
            resized_file = f"data_lib/00_info/{resized_folder}.json"
            
            cond1 = os.path.isfile(resized_file) and os.path.exists(
                f"{self.root}/{resized_folder}")
            cond2 = self.args.experiment.dataset.tar
            if cond1 or cond2:
                self.info_dict = json.load(open(resized_file, "r"))
            else:
                self._create_and_use_resized_data()
                
            self.set_data_and_targets()   
            
        # If needed, preload images from tars instead!
        if self.args.experiment.dataset.tar:
            print(f"Loading {self.root.split('/')[-1]} tar data into memory...")
            tar_path = f"{self.root}.tar"

            # Read the tar file into memory
            with open(tar_path, 'rb') as f:
                tar_data = f.read()

            # Use BytesIO to simulate a file object in memory
            tar_stream = io.BytesIO(tar_data)

            # Open the tar stream with tarfile
            self.preloaded_image_data = [None] * len(self.data)
            idx_hash = {}
            for i, key in enumerate(self.data):
                idx_hash[key.replace(self.args.experiment.dataset.path + '/', '')] = i
            ref_checker = [False] * len(self.data)
            with tarfile.open(fileobj=tar_stream, mode='r:*') as tar:
                # Extract all members
                members = tar.getmembers()
                
                # Extract files into memory
                for member in tqdm.tqdm(members, desc='Getting files...'):
                    # Only extract files (skip directories)
                    if member.isfile():
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            ref_key = member.name
                            if ref_key in idx_hash:
                                ref_file = io.BytesIO(file_obj.read())
                                if self.args.experiment.dataset.full_tar_preload:
                                    ref_file = self._load_image(ref_file)
                                ref_key = idx_hash[ref_key]
                                self.preloaded_image_data[ref_key] = ref_file
                                ref_checker[ref_key] = True
            try:
                assert_str = f"tar-file for {self.root.split('/')[-1]} is incomplete!"
                assert all(ref_checker), assert_str
            except:
                from IPython import embed; embed()    
                            
        # If needed, load data into memory.
        self._preload_data()

    def _download_and_setup(self):
        """Create data and target lists

        Optionally download required files.
        """
        pass

    def _check_for_preprocessed_data(self):
        # CORRECTLY LOAD RESIZED CONTEXT, Captions and targets for caption / anno-data.
        resized_root_folder = f'{self.root}/{self.root.split("/")[-1]}_{self.split}_{self.PARAMS["resize"]}'
        if self.PARAMS["create_resized_variant_if_possible"] and os.path.isdir(
            resized_root_folder
        ):
            self.data_setup_required = False

    def _create_and_use_resized_data(self):
        # For datasets with highly varying and in parts very high image sizes,
        # it is recommend to create a resized copy beforehand to speed up dataloading.
        if self.run_with_resized_data:
            resize_val = self.PARAMS["resize"]

            basename = self.root.split("/")[-1]
            new_folder = f"{basename}_{self.split}_{resize_val}"

            adjusted_folder = f"{self.root}/{new_folder}"
            adjusted_data = [
                f'{new_folder}/{x.split(self.root+"/")[-1]}' for x in self.data
            ]

            if not os.path.isdir(adjusted_folder):
                resizer = torchvision.transforms.Resize(
                    resize_val,
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                )

                os.makedirs(adjusted_folder, exist_ok=True)
                for i, x in tqdm.tqdm(
                    enumerate(self.data),
                    desc="Creating size-adjusted dataset copy...",
                    total=len(self.data),
                ):
                    base_folder = f"{self.root}/{new_folder}"
                    folders = x.split(self.root + "/")[-1].split("/")[:-1]
                    filename = x.split("/")[-1]

                    for folder in folders:
                        base_folder = f"{base_folder}/{folder}"
                        os.makedirs(base_folder, exist_ok=True)

                    img = resizer(Image.open(x).convert("RGB"))
                    new_save_path = f"{base_folder}/{filename}"
                    _ = img.save(new_save_path)

                if not os.path.exists(f"data_lib/00_info/{new_folder}.json"):
                    info_dict = {}
                    for i, key in enumerate(self.info_dict.keys()):
                        info_dict[adjusted_data[i]] = self.info_dict[key]
                    json.dump(
                        info_dict,
                        open(f"data_lib/00_info/{new_folder}.json", "w"),
                        indent=4,
                    )

            self.info_dict = json.load(open(f"data_lib/00_info/{new_folder}.json", "r"))
                    
    def _preload_data(self):
        if self.preload and isinstance(self.data[0], str):
            self.data = [
                np.array(Image.open(x).convert("RGB"))
                for x in tqdm.tqdm(self.data, desc=f"[{self.split}] Preloading data...")
            ]

    def _load_image(self, x):
        if isinstance(x, str) or isinstance(x, pathlib.PosixPath):
            x = Image.open(x).convert("RGB")
        elif isinstance(x, Image.Image):
            pass
        elif isinstance(x, io.BytesIO):
            x = Image.open(x).convert("RGB")
        elif isinstance(x, np.ndarray):
            x = Image.fromarray(x, mode="RGB")
        return x

    def set_params(self):
        if self.args.experiment.dataset.img_size > 0:
            self.PARAMS["img_size"] = self.args.experiment.dataset.img_size
        if self.args.experiment.dataset.resize <= 0:
            self.PARAMS["resize"] = self.PARAMS["img_size"]
        else:
            self.PARAMS["resize"] = self.args.experiment.dataset.resize

    def set_data_and_targets(self):
        self.data, self.targets, self.caption_data = [], [], []
        for key, items in self.info_dict.items():
            self.data.append(f"{self.root}/{key}")
            self.targets.append(items["target"])
            caption = (
                items["default_caption"]
                if items["default_caption"] is not None
                else items["synthetic_merged_caption"]
            )
            if caption is not None:
                self.caption_data.append(caption)

        self.targets = torch.from_numpy(np.array(self.targets)).to(torch.long)

     
    def create_transforms_lists(self):
        self.TRANSFORMS, self.size_adjust, self.normalize = create_transforms_list(
            self.transform_list, self.PARAMS
        )

    def set_transforms(self, transform: torchvision.transforms = None):
        self.create_transforms_lists()
        if transform is None:
            self.transform = torchvision.transforms.Compose(self.TRANSFORMS)
        else:
            self.transform = torchvision.transforms.Compose(transform)
        self.target_transform = None

    def _collate_fn(self, batch: Dict) -> Dict:
        return torch.utils.data.default_collate(batch)
        # collate_out = {}
        # keys = batch[0].keys()
        # for key in keys:
        #     if key == 'non_aug_inputs':
        #         collate_out[key] = [x[key] for x in batch]
        #     else:
        #         collate_out[key] = torch.utils.data.default_collate([x[key] for x in batch])
        # return collate_out

    def __getitem__(self, index: int) -> dict:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: dict.
        """
        # Ensure that we can utilize dataloaders of lengths larger than the available dataset.
        index = index % len(self.data)
        image_path = self.data[index]
        if isinstance(image_path, pathlib.PosixPath):
            image_path = str(image_path)
        if self.preloaded_image_data:
            orig_img = img = self._load_image(self.preloaded_image_data[index])
        else:
            orig_img = img = self._load_image(image_path)
        target = self.targets[index]

        return_dict = {}

        if self.transform is not None:
            base_img = self.transform(img)

            if self.train and self.req_aux_inputs:
                aux_img = self.transform(img)
                return_dict["aux_inputs"] = aux_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return_dict["indices"] = index
        return_dict["images"] = base_img
        return_dict["image_path"] = image_path
        return_dict["targets"] = target

        # For caption-only datasets, we do not pass along any classnames.
        if self.PARAMS["base_classes"] is not None:
            return_dict["classnames"] = self.PARAMS["base_classes"][target]
            return_dict["primed_classes"] = self.PARAMS["classes"][target]

        # If caption-data is available or not use primer captions, we pass it through 'texts'.
        # If no caption-data is available, 'texts' will be filled by classnames.
        if self.caption_data is None or (
            self.PARAMS["classes"] is not None and
            self.args.experiment.dataset.primer_captions
        ):
            return_dict["texts"] = self.PARAMS["classes"][target]
        else:
            # caption_key = index
            # if isinstance(image_path, str):
            #     caption_key = image_path.replace('/'.join(self.root.split('/')[:-1]) + '/', '')
            #     if caption_key[:5] == 'data/':
            #         caption_key = caption_key[5:]
            return_dict["texts"] = self.caption_data[index]

            if isinstance(return_dict["texts"], list):
                return_dict["texts"] = (
                    return_dict["texts"][0]
                    if len(return_dict["texts"]) == 1 or not self.train
                    else np.random.choice(return_dict["texts"])
                )

        # If the model requires non-augmented image input data, we augment the item-list accordingly.
        if self.train and self.req_non_aug_inputs:
            # return_dict['non_aug_inputs'] = self.to_tensor(self.size_adjust(orig_img))
            return_dict["non_aug_inputs"] = self.to_tensor(orig_img)

        if hasattr(self, "logits"):
            return_dict["logits"] = self.logits[index]

        return return_dict

    def __len__(self):
        return len(self.data)

def create_transforms_list(
    transform_list: List[str], PARAMS: Dict, return_list_only: bool = True
):
    interpolation_mode = torchvision.transforms.InterpolationMode.BICUBIC

    TRANSFORMS = []

    size_adjust = lambda x: x

    allowed_transforms = [
        "Resize",
        "RandomResizedCrop",
        "RandomCropWithPadding",
        "CenterCrop",
        "RandomHorizontalFlip",
        "ToTensor",
        "Normalize",
    ]

    for transform_name in transform_list:
        assert_str = f"Transform [{transform_name}] not available. Please choose from {allowed_transforms}!"
        assert transform_name in allowed_transforms, assert_str

    if any(["Resize" in x for x in transform_list]):
        size_adjust = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    PARAMS["resize"],
                    interpolation=interpolation_mode,
                    max_size=None,
                    antialias=None,
                ),
                torchvision.transforms.CenterCrop(PARAMS["img_size"]),
            ]
        )

    if "Resize" in transform_list:
        TRANSFORMS.append(
            torchvision.transforms.Resize(
                PARAMS["resize"],
                interpolation=interpolation_mode,
                max_size=None,
                antialias=None,
            )
        )

    if "RandomResizedCrop" in transform_list:
        TRANSFORMS.append(
            torchvision.transforms.RandomResizedCrop(
                PARAMS["img_size"],
                interpolation=interpolation_mode,
                scale=(0.9, 1.0),
                ratio=(0.75, 1.3333),
            )
        )

    if "RandomCropWithPadding" in transform_list:
        TRANSFORMS.append(
            torchvision.transforms.RandomCrop(PARAMS["img_size"], padding=4)
        )

    if "CenterCrop" in transform_list:
        TRANSFORMS.append(torchvision.transforms.CenterCrop(PARAMS["img_size"]))
    if "RandomHorizontalFlip" in transform_list:
        TRANSFORMS.append(torchvision.transforms.RandomHorizontalFlip())
    if "ToTensor" in transform_list:
        TRANSFORMS.append(torchvision.transforms.ToTensor())

    normalize = lambda x: x
    if "Normalize" in transform_list:
        normalize = torchvision.transforms.Normalize(PARAMS["mean"], PARAMS["std"])
        TRANSFORMS.append(normalize)

    if return_list_only:
        return TRANSFORMS, size_adjust, normalize
    else:
        return TRANSFORMS, size_adjust, normalize


def get_single_dataset(
    args, train_transform=None, test_transform=None, target_transform=None
) -> torch.utils.data.Dataset:
    name = args.experiment.dataset.name
    assert name in DATASETS.keys()
    mod = importlib.import_module(DATASETS[name])

    root = args.experiment.dataset.path
    train_dataset = getattr(mod, "Dataset")(
        root=root,
        train=True,
        transform=train_transform,
        target_transform=target_transform,
    )
    test_dataset = getattr(mod, "Dataset")(
        root=root,
        train=False,
        transform=test_transform,
        target_transform=target_transform,
    )

    with open_dict(args):
        args.experiment.dataset.classes = train_dataset.PARAMS["classes"]
        args.experiment.dataset.num_classes = train_dataset.PARAMS["num_classes"]
        args.experiment.dataset.mean = train_dataset.PARAMS["mean"]
        args.experiment.dataset.std = train_dataset.PARAMS["std"]

    return {"train": train_dataset, "test": test_dataset}


def get_train_val_split_indices(
    targets: Union[List, np.ndarray, torch.Tensor], train_val_split: float
):
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy().tolist()
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()

    tar_coll = {}
    for i, tar in enumerate(targets):
        if tar not in tar_coll:
            tar_coll[tar] = []
        tar_coll[tar].append(i)

    train_idcs_coll = {tar: None for tar in tar_coll.keys()}
    val_idcs_coll = {tar: None for tar in tar_coll.keys()}

    for tar, values in tar_coll.items():
        split_val = int(len(values) * train_val_split)
        train_idcs_coll[tar] = list(np.random.choice(values, split_val, replace=False))
        val_idcs_coll[tar] = list(set(values) - set(train_idcs_coll[tar]))

    train_idcs = train_idcs_coll.values()
    train_idcs = sorted([x for y in train_idcs for x in y])
    val_idcs = val_idcs_coll.values()
    val_idcs = sorted([x for y in val_idcs for x in y])

    return np.array(train_idcs).astype(int), np.array(val_idcs).astype(int)

def get_datasets(
    args,
    train_transform=["ToTensor", "Normalize"],
    test_transform=["ToTensor", "Normalize"],
    target_transform=None,
) -> torch.utils.data.Dataset:
    names = args.experiment.dataset.name
    for name in names:
        assert name in DATASETS.keys(), (
            f"Dataset {name} not available. Please choose from {list(DATASETS.keys())}"
        )

    eval_names = args.experiment.evaluation.additional_datasets
    for name in eval_names:
        assert name in DATASETS.keys(), (
            f"Dataset {name} not available. Please choose from {list(DATASETS.keys())}"
        )

    train_datasets = []
    test_datasets = []
    eval_only_test_datasets = []

    PARAMS_COLL = {"classes": [], "num_classes": [], "mean": [], "std": []}

    for name in names:
        start = time.time()
        print(f"[ADAPT] Preparing {name}...")
        mod = importlib.import_module(DATASETS[name])

        preload = args.experiment.dataset.preload
        if args.experiment.dataset.preload_test_only:
            preload = False

        train_dataset = getattr(mod, "Dataset")(
            root=args.experiment.dataset.path,
            args=args,
            train=True,
            transform=train_transform,
            target_transform=target_transform,
            preload=preload,
            create_resized_data=args.experiment.dataset.create_resized_variant_if_possible,
            img_size=args.experiment.dataset.img_size,
        )

        if args.experiment.dataset.preload_test_only:
            preload = True

        if not args.experiment.dataset.validation_mode:
            train_datasets.append(train_dataset)
            test_datasets.append(
                getattr(mod, "Dataset")(
                    root=args.experiment.dataset.path,
                    args=args,
                    train=False,
                    transform=test_transform,
                    target_transform=target_transform,
                    preload=preload,
                    create_resized_data=args.experiment.dataset.create_resized_variant_if_possible,
                )
            )
        else:
            train_idcs, val_idcs = get_train_val_split_indices(
                train_dataset.targets, args.experiment.dataset.train_val_split
            )
            val_dataset = getattr(mod, "Dataset")(
                root=args.experiment.dataset.path,
                args=args,
                train=True,
                transform=train_transform,
                target_transform=target_transform,
                preload=preload,
                create_resized_data=args.experiment.dataset.create_resized_variant_if_possible,
            )
            if isinstance(train_dataset.data, np.ndarray) or isinstance(
                train_dataset.data, torch.Tensor
            ):
                train_dataset.data = train_dataset.data[train_idcs]
                val_dataset.data = val_dataset.data[val_idcs]
            else:
                train_dataset.data = [
                    x for i, x in enumerate(train_dataset.data) if i in train_idcs
                ]
                val_dataset.data = [
                    x for i, x in enumerate(val_dataset.data) if i in val_idcs
                ]

            if isinstance(train_dataset.targets, np.ndarray) or isinstance(
                train_dataset.targets, torch.Tensor
            ):
                train_dataset.targets = train_dataset.targets[train_idcs]
                val_dataset.targets = val_dataset.targets[val_idcs]
            else:
                train_dataset.targets = [
                    x for i, x in enumerate(train_dataset.targets) if i in train_idcs
                ]
                val_dataset.targets = [
                    x for i, x in enumerate(val_dataset.targets) if i in val_idcs
                ]

            train_datasets.append(train_dataset)
            test_datasets.append(val_dataset)

        PARAMS_COLL["classes"].append(train_datasets[-1].PARAMS["classes"])
        PARAMS_COLL["num_classes"].append(train_datasets[-1].PARAMS["num_classes"])
        PARAMS_COLL["mean"].append(train_datasets[-1].PARAMS["mean"])
        PARAMS_COLL["std"].append(train_datasets[-1].PARAMS["std"])

        print("- Done in {0:4.1f}s.".format(time.time() - start))
        print(
            "- Train: {0} | Test: {1}".format(
                len(train_datasets[-1]), len(test_datasets[-1])
            )
        )
        print("- Num. Classes: {0}".format(len(np.unique(train_datasets[-1].targets))))

    for name in eval_names:
        start = time.time()

        mod = importlib.import_module(DATASETS[name])

        # To avoid test data leakage for hyperparameter tuning, validation_mode = True will
        # simulate evaluation-only data through validation splits from ContinualFoMo training data!
        if not args.experiment.dataset.validation_mode:
            print(f"[EVAL ONLY] Preparing {name}...")
            eval_only_test_datasets.append(
                getattr(mod, "Dataset")(
                    root=args.experiment.dataset.path,
                    args=args,
                    train=False,
                    transform=test_transform,
                    target_transform=target_transform,
                    preload=args.experiment.dataset.preload,
                    create_resized_data=args.experiment.dataset.create_resized_variant_if_possible,
                )
            )
        else:
            print(f"[EVAL ONLY, but validation split] Preparing {name}...")
            # We have to load the associated training split to create a validation dataset from.
            eval_dataset = getattr(mod, "Dataset")(
                root=args.experiment.dataset.path,
                args=args,
                train=True,
                transform=test_transform,
                target_transform=target_transform,
                preload=False,
                create_resized_data=args.experiment.dataset.create_resized_variant_if_possible,
            )
            # Create validation split for evaluation-only data.
            _, val_idcs = get_train_val_split_indices(
                eval_dataset.targets, args.experiment.dataset.train_val_split
            )
            if isinstance(eval_dataset.data, np.ndarray) or isinstance(
                eval_dataset.data, torch.Tensor
            ):
                eval_dataset.data = eval_dataset.data[val_idcs]
            else:
                eval_dataset.data = [
                    x for i, x in enumerate(eval_dataset.data) if i in val_idcs
                ]

            if isinstance(eval_dataset.targets, np.ndarray) or isinstance(
                eval_dataset.targets, torch.Tensor
            ):
                eval_dataset.targets = eval_dataset.targets[val_idcs]
            else:
                eval_dataset.targets = [
                    x for i, x in enumerate(eval_dataset.targets) if i in val_idcs
                ]

            eval_only_test_datasets.append(eval_dataset)

        PARAMS_COLL["classes"].append(eval_only_test_datasets[-1].PARAMS["classes"])
        PARAMS_COLL["num_classes"].append(
            eval_only_test_datasets[-1].PARAMS["num_classes"]
        )
        PARAMS_COLL["mean"].append(eval_only_test_datasets[-1].PARAMS["mean"])
        PARAMS_COLL["std"].append(eval_only_test_datasets[-1].PARAMS["std"])

        print("- Done in {0:4.1f}s.".format(time.time() - start))
        print("- Eval Samples: {0}".format(len(eval_only_test_datasets[-1])))
        print(
            "- Num. Classes: {0}".format(
                len(np.unique(eval_only_test_datasets[-1].targets))
            )
        )

    with open_dict(args):
        args.experiment.dataset.classes = PARAMS_COLL["classes"]
        args.experiment.dataset.num_classes = PARAMS_COLL["num_classes"]
        args.experiment.dataset.mean = PARAMS_COLL["mean"]
        args.experiment.dataset.std = PARAMS_COLL["std"]

    return {
        "train": train_datasets,
        "test": test_datasets,
        "eval_only_test": eval_only_test_datasets,
    }


def summarize(args: omegaconf.DictConfig, datasets_dict: dict):
    summary_str = ""
    for i, (key, dataset) in enumerate(datasets_dict.items()):
        num_samples = len(dataset)
        if isinstance(dataset, list):
            num_samples = sum([len(x) for x in dataset])
        num_classes = sum(
            [
                x.PARAMS["num_classes"]
                for x in dataset
                if x.PARAMS["num_classes"] is not None
            ]
        )
        num_retrieval_datasets = sum([x.PARAMS["type"] == "retrieval" for x in dataset])
        summary_str += f"- {key}: {num_samples} samples [{num_classes} classes, {num_retrieval_datasets} retrieval task(s)]."
        if i < len(datasets_dict) - 1:
            summary_str += "\n"
    termcolor.cprint("\nOverall Data Summary:", "white", attrs=["underline"])
    if args.experiment.dataset.sequence is not None:
        termcolor.cprint("Note: These are full dataset sizes. As custom sequence is given:\nPlease check experiment summary for exact numbers.", "yellow")
    print(summary_str)
    termcolor.cprint("\nDatasets utilized:", "white", attrs=["underline"])
    print(" | ".join(args.experiment.dataset.name))

def get_datacomp_loader(
    args,
    train_transform=["RandomResizedCrop", "ToTensor", "Normalize"],
    train_batch_size=512,
    custom_seed=None
):

    ### Get DataComp tar indexes to pass to web-loader
    pretraining_root = args.experiment.dataset.pretraining_data_path
    archives = None
    try:
        archives = sorted([x for x in os.listdir(pretraining_root) if '.tar' in x])
    except Exception as e:
        raise ValueError('Unable to load tars from {}, exception: {}'.format(pretraining_root, str(e)))
    assert archives is not None, '{} does not contain any tar files for DataComp loading.'.format(pretraining_root)

    use_inds_ = [archives[0].split('.')[0], archives[-1].split('.')[0]]
    use_inds_tar_path = os.path.join(pretraining_root, '{'+use_inds_[0]+'..'+use_inds_[1]+'}.tar')

    ## Get transforms list
    params_ = {
        "img_size": 224, # default image-size
        "resize": 224,
        "mean": OPENAI_DATASET_MEAN,
        "std": OPENAI_DATASET_STD,
    }
    tfs, _, _ = create_transforms_list(train_transform, params_)
    tfs = torchvision.transforms.Compose(tfs)

    return get_wds_dataset(
        use_inds_tar_path,
        preprocess_img=tfs,
        preprocess_label=lambda x: x,
        is_train=True,
        epoch=0, # harcode this to get deterministic orderings over the datacomp samples
        batch_size=train_batch_size,
        num_samples=DATACOMP_SIZE_,
        resampled=False, # to get deterministic orderings over the datacomp samples across runs
        seed=args.experiment.seed if not custom_seed else custom_seed,
        workers=args.experiment.dataset.num_workers,
    )


ConversionDataHandle = {
    'AI2DIAGRAMS':'ai2diagrams',
    'ARTBENCH10':'artbench10',
    'BIRDSNAP':'birdsnap',
    'CALTECH101':'caltech101',
    'CALTECH256':'caltech256',
    'CARS196':'cars196',
    'CIFAR10':'cifar10',
    'CIFAR100':'cifar100',
    'CLEVR':'clevr',
    'CLRS':'clrs',
    'COUNTRY211':'country211',
    'CUB200':'cub200',
    'DF20MINI':'df20mini',
    'DOLLARSTREET':'dollarstreet',
    'DOMAINNET_CLIPART':'domainnet_clipart',
    'DOMAINNET_INFOGRAPH':'domainnet_infograph',
    'DOMAINNET_PAINTING':'domainnet_painting',
    'DOMAINNET_QUICKDRAW':'domainnet_quickdraw',
    'DOMAINNET_SKETCH':'domainnet_sketch',
    'DSPRITES':'dsprites',
    'DTD':'dtd',
    'EUROSAT':'eurosat',
    'EuroSAT':'eurosat',
    'FashionMNIST':'fashionmnist',
    'FGVCAIRCRAFT':'fgvcaircraft',
    'FGVCAircraft':'fgvcaircraft',
    'FLICKR30K':'flickr30k',
    'FLOWERS102':'flowers102',
    'FOOD101':'food101',
    'Food101':'food101',
    'FRU92':'fru92',
    'FSCOCO':'fscoco',
    'GTSRB':'gtsrb',
    'IMAGENET':'imagenet',
    'IMAGENET_A':'imagenet_a',
    'IMAGENET_D':'imagenet_d',
    'IMAGENET_R':'imagenet_r',
    'IMAGENET_S':'imagenet_s',
    'IMAGENET_V2':'imagenet_v2',
    'iNATURALIST2021':'inaturalist2021',
    'ISICMELANOMA':'isicmelanoma',
    'MedMNISTderma':'medmnist_derma',
    'MedMNISTorganc':'medmnist_organc',
    'MedMNISTorgans':'medmnist_organs',
    'MITSTATES':'mitstates',
    'MNIST':'mnist',
    'MONKEYS10':'monkeys10',
    'MSCOCO':'mscoco',
    'MTSD':'mtsd',
    'MVTECAD_Adapt':'mvtecad_adapt',
    'MVTECAD_Eval':'mvtecad_eval',
    'OBJECTNET':'objectnet',
    'OBSC_ANIMALS':'obsc_animals',
    'OBSC_THINGS':'obsc_things',
    'OPENIMAGES':'openimages',
    'OXFORDPETS':'oxford_pets',
    'PATTERNNET':'patternnet',
    'PLACES365':'places365',
    'PLANTVILLAGE':'plantvillage',
    'QUILT':'quilt',
    'RESISC45':'resisc45',
    'RETINOPATHY':'retinopathy',
    'SHAPES3D':'shapes3d',
    'SNAKECLEF':'snake_clef',
    'STL10':'stl10',
    'SUN397':'sun397',
    'SVHN':'svhn',
    'SynthCLIP106':'synthclip106',
    'VEG200':'veg200',
    'ZAPPOS50k':'zappos50k'   
}

ConversionHandleData = {value: key for key, value in ConversionDataHandle.items()}