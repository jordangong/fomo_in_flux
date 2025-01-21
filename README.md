# The Practitioner's Guide to Continual Multimodal Pretraining

Short version published at ___NeurIPS 2024 (Dataset and Benchmarks Track)___ and at the ___NeurIPS 2024 ContinualFomo Workshop (Oral)___.

-----------------------------

__:pen: Authors:__
Karsten Roth$^*$, Vishaal Udandarao$^*$, Sebastian Dziadzio$^\circ$, Ameya Prabhu$^\circ$, Mehdi Cherti, Oriol Vinyals, Olivier J. Henaff, Samuel Albanie$^\dagger$, Matthias Bethge$^\dagger$, Zeynep Akata$^\dagger$.  
$^*$equal interchangeable, $^\circ$core contributor, $^\dagger$ shared senior

__:eyes: Summary:__

![Fomo-in-Flux Setup Visualization](images/setup.png)

This repository contains the complete __Fomo-in-Flux (FiF)__ pipeline introduced in [our work](https://arxiv.org/abs/2408.14471) (NeurIPS Dataset & Benchmark track 2024) and used in our [follow-up work on temporal model merging](https://arxiv.org/abs/2412.06712), and contains the following components:

* Entire __FiF__ pipeline from precise data streaming to cost budgeting and evaluation.
* Configurations and mostly complete automatic download links to most of the 63 datasets used for adaptation and evaluation. More information in the respective data section.
* Implementation of parameter-additive and -selective update methods, standard continual learning mechanisms, finetuning techniques and model merging approaches.
* Precomputed data streams mimicking different deployment scenarios described in our paper.
* An extensive evaluation on 21 different datasets.
* Result logging using Weights & Biases.

__Moreover__, this codebase is easily used as a general powerful continual pretraining pipeline to study multimodal representation models.

-----------------------------

__:exclamation: Some disclaimer:__  

* If you find any licensing issues with our redistribution links, please let us know, and we will take it down immediately and replace it with suitable preprocessing scripts!
* Due to licensing and hosting issues, we were not able to provide preprocessed download links for some of the pretraining datasets (particularly the pretraining replay data).
* As a 1-to-1 copy of our utilized data streams (particular w.r.t. to pretraining datasets) is not possible, slight numerical variance may be encountered during replication.
* Exact checkpoint replication is currently not possible; i.e. if a run is interrupted and resumed, results will not be the same as its uninterrupted equivalent. However, two uninterrupted runs should return the same values.

__:fire: Version History:__

* Uploaded initial version 1.0. This contains the full Fomo-in-Flux pipeline. Next change will fix some checkpointing issues currently encountered.

__:book: Citation__
If you found this repository useful for your research, please consider citing it via

```bibtex
@article{roth2024practitioner,
  title={A Practitioner's Guide to Continual Multimodal Pretraining},
  author={Roth, Karsten and Udandarao, Vishaal and Dziadzio, Sebastian and Prabhu, Ameya and Cherti, Mehdi and Vinyals, Oriol and H{\'e}naff, Olivier and Albanie, Samuel and Bethge, Matthias and Akata, Zeynep},
  journal={arXiv preprint arXiv:2408.14471},
  year={2024}
}
```

-----------------------------

## Table of Contents

- [The Practitioner's Guide to Continual Multimodal Pretraining](#the-practitioners-guide-to-continual-multimodal-pretraining)
  - [Table of Contents](#table-of-contents)
  - [1. Requirements](#1-requirements)
    - [1.1 Setting up the Fomo-in-Flux environment](#11-setting-up-the-fomo-in-flux-environment)
    - [1.2 Data Preparation](#12-data-preparation)
  - [2. Getting Started](#2-getting-started)
    - [2.1 Quick Verifications](#21-quick-verifications)
      - [2.1.1 Summarize Downloaded Datasets - Visual Inspection](#211-summarize-downloaded-datasets---visual-inspection)
      - [2.1.2 Zeroshot Evaluation](#212-zeroshot-evaluation)
    - [2.2 Starting your first FiF-experiments](#22-starting-your-first-fif-experiments)
      - [2.2.1 A first full, but short FiF verification run](#221-a-first-full-but-short-fif-verification-run)
      - [2.2.2 Continual Pretraining of ViT-B/16 on full FiF](#222-continual-pretraining-of-vit-b16-on-full-fif)
  - [3. Key settings and flags you may want to modify](#3-key-settings-and-flags-you-may-want-to-modify)
    - [3.1 Defining your experiment config file](#31-defining-your-experiment-config-file)
    - [3.2 Determining Compute Budgets: Profile \& Count GFlops](#32-determining-compute-budgets-profile--count-gflops)
    - [3.3 Fomo-in-Flux training: Key parameters](#33-fomo-in-flux-training-key-parameters)
    - [3.4 Running on data tar files](#34-running-on-data-tar-files)
    - [3.5 Multi-GPU and RAM preloading](#35-multi-gpu-and-ram-preloading)
    - [3.6 Adding your own twist to FiF](#36-adding-your-own-twist-to-fif)
      - [3.6.1 Custom dataset](#361-custom-dataset)
      - [3.6.2 Custom continual learner](#362-custom-continual-learner)
      - [3.6.3 Custom backbone](#363-custom-backbone)
    - [3.7 Running on validation splits](#37-running-on-validation-splits)
    - [3.8 Improving evaluation speeds during streaming](#38-improving-evaluation-speeds-during-streaming)
    - [3.9 Evaluation from W\&B xid](#39-evaluation-from-wb-xid)
    - [3.10 W\&B logging and checkpointing](#310-wb-logging-and-checkpointing)
  
-----------------------------

## 1. Requirements

### 1.1 Setting up the Fomo-in-Flux environment

To utilize this repository, simply set up a corresponding environment using

```bash
conda env create -f environment.yaml
```

and activate the environment via `conda activate fif`. This should install all required libraries.

### 1.2 Data Preparation

This part is the most time-consuming, as all 63 datasets have to be downloaded and preprocessed. Fortunately, we currently provide download links for preprocessed dataset variants for __all__ adaptation and evaluation datasets. These can be downloaded using:

```bash
bash download_and_process_all_datasets.sh path/to/download/dir  # e.g. 'data'
```

This will download preprocessed dataset `tar`-files and extract them. The entire download takes about 2.5 hours with a standard 50 Mbps download speed; the total size is around 65 GB.

__Moreover__, note that while the data is extracted by default, it is also possible to conduct __FiF__-studies on the dataset tar-files directly in scenarios where total filecount is bounded (albeit slightly slower).

__Note:__
All modified datasets are provided under the same license as the originals with references below. Users are responsible for adhering to the terms of the original licenses. For reference, links to the original datasets and their licenses are included in this repository wherever possible. Any modifications, preprocessing, or enhancements applied to these datasets do not imply endorsement by the original dataset creators. When using these datasets, please ensure to reference them accordingly!

__Importantly__, these datasets have been preprocessed particularly for use within Fomo-in-Flux, and do not serve as a stand-in for the original datasets in other applications.

Here is a table of all datasets used, and the respective references and licenses we were able to verify (dataset names reference the downloaded folder names):

<details>

We now summarize all downloadable datasets. Please reference [the paper](https://arxiv.org/pdf/2408.14471) for the respective exact references!

__Adaptation Datasets:__

![Adaptation Datasets](images/train_datasets.png)

__Evaluation Datasets:__

![Evaluation Datasets](images/eval_datasets.png)

__Association Table:__

| Download Name   | Table Name |
|-------------|-------------|
| AI2DIAGRAMS | AI2Diagrams|
| ARTBENCH10 | ArtBench10|
| BIRDSNAP | Birdsnap|
| CALTECH101 | Caltech101|
| CALTECH256 | Caltech256|
| CARS196 | Cars196|
| CIFAR10 | Cifar10|
| CIFAR100 | Cifar100|
| CLEVR | CLEVR|
| CLRS | CLRS|
| COUNTRY211 | Country211|
| CUB200 | CUB200-2011|
| DF20MINI | DF20-mini|
| DOLLARSTREET | Dollarstreet|
| DOMAINNET_CLIPART | Domainnet-Clipart|
| DOMAINNET_INFOGRAPH | Domainnet-Infograph|
| DOMAINNET_PAINTING | Domainnet-Painting|
| DOMAINNET_QUICKDRAW | Domainnet-Quickdraw|
| DOMAINNET_SKETCH | Domainnet-Sketch|
| DSPRITES | Dsprites|
| EuroSAT | EuroSAT|
| FashionMNIST | FashionMNIST|
| FGVCAircraft | FGVCAircraft|
| FLICKR30K | Flickr300k|
| FLOWERS102 | Flowers102|
| Food101 | Food101|
| FRU92 | FRU92|
| FSCOCO | FSCOCO|
| GTSRB | GTSRB|
| IMAGENET | ImageNet (val only)|
| IMAGENET_A | ImageNet-A|
| IMAGENET_D | ImageNet-D|
| IMAGENET_R | ImageNet-R|
| IMAGENET_S | ImageNet-S|
| IMAGENET_V2 | ImageNet-V2|
| iNATURALIST2021 | iNaturalist2021|
| ISICMELANOMA | Isicmelanoma|
| MITSTATES | Mitstates|
| MNIST | MNIST|
| MONKEYS10 | Monkeys10|
| MSCOCO | MSCOC|
| MTSD | MTSD|
| MVTECAD_Adapt | MVTEC-AD (Base)|
| MVTECAD_Eval | MVTEC-AD (Faults)|
| OBJECTNET | ObjectNet|
| OBSC_ANIMALS | Obscure Animals|
| OBSC_THINGS | Obscure Things|
| OPENIMAGES | OpenImages|
| OXFORDPETS | OxfordPets|
| PATTERNNET | PatternNet|
| PLACES365 | Places365|
| PLANTVILLAGE | Plantvillage|
| QUILT | Quilt-1M|
| RESISC45 | Resisc45|
| SHAPES3D | Shapes3D|
| SNAKECLEF | SnakeCLEF2023|
| STL10 | STL10|
| SUN397 | SUN397|
| SVHN | SVHN|
| SynthCLIP106 | SynthCLIP106|
| VEG200 | Veg200|
| ZAPPOS50k | Zappos50k|

</details>

</br>

__Datasets for manual download:__

Currently, the only data required to be manually download, is data associated with the original pretraining data:

In practical scenarios, particularly when deploying larger models, one should have access to pretraining data as well for replay during continual pretraining. We study different re-pretraining datapools in __FiF__ as well, by replaying e.g. on 2M sample subsets from `LAION-400M`, `CC-12M`, `CC-3M` or `DataComp-Small`. Respective chunks can be found on the respective hosting website (c.f. e.g https://laion.ai/blog/laion-400-open-dataset/).

We also provide download scripts in `download_laion400m` and `download_datacomp` for ease of use.

-----------------------------

## 2. Getting Started

### 2.1 Quick Verifications

#### 2.1.1 Summarize Downloaded Datasets - Visual Inspection

Once the data is downloaded, you can run:

```bash
python summarize_codebase.py
```

to verify the data. It will create exemplary image-caption dataset visualizations in the `dataset_visualizations` folder (for both `adapt/train` and `eval/test` splits).

#### 2.1.2 Zeroshot Evaluation

Next, you can run:

```bash
python main.py experiment=zeroshot_evaluation_complete.yaml experiment.backbone.name=openclip_vit_b32 experiment.backbone.cache_dir=/path/to/download/clip/checkpoints log.folder=/path/to/save/logs/
```

to run all zero-shot evaluations. Then you can verify these numbers with the numbers here:

<details>

__OpenCLIP ViT-B/32 Verification Zero-Shot Numbers:__

|Dataset Name (per `data_lib` name) | ViT-B/32 ZS Score |
|--------|--------|
|`ai2diagrams.py` | 72.7 |
|`artbench10.py` | 15.0 |
|`birdsnap.py` | 46.7 |
|`caltech101.py` | 90.7 |
|`caltech256.py` | 88.5 |
|`cars196.py` | 85.9 |
|`cifar100.py` | 75.8 |
|`cifar10.py` | 93.7 |
|`clevr.py` | 4.7 |
|`clrs.py` | 59.7 |
|`country211.py` | 14.6 |
|`cub200.py` | 64.1 |
|`df20mini.py` | 2.1 |
|`dollarstreet.py` | 5.5 |
|`domainnet_clipart.py` | 78.7 |
|`domainnet_infograph.py` | 50.9 |
|`domainnet_painting.py` | 71.9 |
|`domainnet_quickdraw.py` | 19.1 |
|`domainnet_sketch.py` | 68.9 |
|`dsprites.py` | 11.8 |
|`dtd.py` | 54.5 |
|`eurosat.py` | 49.2 |
|`fashionmnist.py` | 77.0 |
|`fgvcaircraft.py` | 23.9 |
|`flowers102.py` | 68.0 |
|`food101.py` | 82.5 |
|`fru92.py` | 50.3 |
|`gtsrb.py` | 48.6 |
|`imagenet.py` | 65.5 |
|`imagenet_a.py` | 23.1 |
|`imagenet_d.py` | 40.7 |
|`imagenet_r.py` | 73.9 |
|`imagenet_s.py` | 52.3 |
|`imagenet_v2.py` | 57.6 |
|`inaturalist2021.py` | 5.4 |
|`isicmelanoma.py` | 14.2 |
|`mitstates.py` | 25.6 |
|`mnist.py` | 69.2 |
|`monkeys10.py` | 82.7 |
|`mtsd.py` | 20.2 |
|`mvtecad_adapt.py` | 80.2 |
|`mvtecad_eval.py` | 16.8 |
|`objectnet.py` | 32.1 |
|`obsc_animals.py` | 58.9 |
|`obsc_things.py` | 54.0 |
|`openimages.py` | 49.4 |
|`oxford_pets.py` | 90.7 |
|`patternnet.py` | 63.6 |
|`places365.py` | 41.8 |
|`plantvillage.py` | 28.3 |
|`quilt.py` | 0.2 |
|`resisc45.py` | 62.5 |
|`shapes3d.py` | 16.8 |
|`snake_clef.py` | 0.3 |
|`sun397.py` | 68.2 |
|`stl10.py` | 96.4 |
|`svhn.py` | 42.8 |
|`synthclip106.py` | 41.0 |
|`veg200.py` | 32.5 |
|`zappos50k.py` | 17.5 |
|`mscoco.py` | 72.0 |
|`flickr30k.py` | 92.3 |
|`fscoco.py` | 8.1 |

</details>

</br>

For a quick __single-dataset__ evaluation, simply run e.g.

```bash
python main.py experiment=zeroshot_evaluation_complete.yaml experiment.backbone.name=openclip_vit_b32 experiment.dataset.name=['clevr'] experiment.evaluation.additional_datasets=[] experiment.backbone.cache_dir=/path/to/download/clip/checkpoints log.folder=/path/to/save/logs/
```

### 2.2 Starting your first FiF-experiments

#### 2.2.1 A first full, but short FiF verification run

To test if __all__ pipeline elements, including replay from pretraining shards, is working correctly without starting a full run, simple run

```bash
python main.py experiment=continualfomo_debug.yaml continual.method=ema_paint experiment.backbone.name=openclip_vit_b32 log.checkpoint=False experiment.task.batch_size=256 log.name=example_fif_debug_run log.use=False
```

__Note:__

* Set `log.use=False` if you want to test without `wandb`-logging. Otherwise, make sure to set `log.wandb_key` to the correct api key.
* Please set `experiment.backbone.cache_dir=<your_cache_dir> log.folder=<your_log_dir>` depending on your respective openclip/hf cache directory and the desired logging folder. If left to default, everything will be stored in the main operating folder.
* In `experiment.datasets.pretraining_data_path` you point to the subset of the original pretraining data you wish to use, e.g. `laion400m/shards`.

#### 2.2.2 Continual Pretraining of ViT-B/16 on full FiF

Example run conducting continual finetuning on the entire FiF-benchmark with respective `10%`-subset evaluation during training and full evaluation at the end of streaming:

```bash
python main.py experiment=continualfomo_default.yaml continual.method=finetune experiment.backbone.name=openclip_vit_b32 log.checkpoint=False experiment.task.batch_size=256 log.name=example_fif_full_run_with_finetune log.use=False
```

By default, this mixes pretraining, replay and streaming data equally. If you do not want to replay on buffer and pretraining data, simply set

```bash
experiment.dataset.data_mixture.pretraining=0 experiment.dataset.data_mixture.buffer=0 experiment.dataset.data_mixture.update=1.0
```

__Variations to this run:__

[1] For a `rank=64`-LoRA, simply define: `continual.method=lora continual.lora.rank=64`. Moreover, make sure to correctly set the number of allowed samples per task in `experiment.task.n_samples`.

[2] Task-arithmetic model merging:

Make sure to include:

```bash
continual.ema_paint.backbone_merge.method=task_arithmetic continual.ema_paint.head_merge.method=task_arithmetic continual.ema_paint.backbone_merge.apply_lines=False continual.ema_paint.head_merge.apply_lines=False continual.ema_paint.backbone_merge.lines_params=[0.5,0.5,True,'linear'] continual.ema_paint.head_merge.lines_params=[0.5,0.5,True,'linear']
```

Again, ensure that the number of update steps per task (as in `experiment.task.n_samples=<gflops_samples>`) are set based on the gflops required for a forward + backward pass (for the how-to, see the respective section in this readme). Note that for most method setups, Tab. 4 in [our paper](https://arxiv.org/pdf/2408.14471) contains respective GFlops (also useful for verification) and number of steps (i.e. `experiment.task.n_samples` / `experiment.task.batch_size`). The table is also copied here for easy access:  

<details>
![GFLOPS Table](images/gflops.png)
</details>

</br>

In general, each method used is defined using its hyperparameters in `configs/config.yaml/continual.<method_name>`.

-----------------------------

## 3. Key settings and flags you may want to modify

This section provides a detailed breakdown of relevant flags you may wish to set and alter, as well as information on how to best add new backbones, methods or data streams. If there are any questions, please don't hesitate to raise an issue ;).

### 3.1 Defining your experiment config file

___Two ways to stream___

In this codebase, there are two ways to stream your data:

1. __Defining a streaming sequence__: As shown in the available streaming files in `stream_construction`, you can simply define a `.json`-file that contains a list with elements of the form `dataset_name+++classname`. This will take each class from the respective datasets to create a respective stream. By defining `experiment.task.num`, the stream will then get chunked into a respective number of tasks. This means that through the sequence of datasets and classes you choose to introduce into the sequence, you can define arbitrary semantic streaming orderings! Note that you don't have to include all dataset classes (but can to stream the entire dataset), but can simply include whatever class from whichever dataset you want. The pipeline handles all the rest: Creating of the respective streaming and task dataloaders. After defining the `name_of_stream.json` file, simply set `experiment.data.sequence=stream_construction/name_of_stream.json`, and `experiment.data.name=[]`.

2. __Defining streaming datasets__: Vice versa, you can also choose to forego the creation of explicit streaming sequences. Instead, you can simply pass the name of the datasets you want to stream over via `experiment.data.name=[list_of_dataset_names]` (setting `experiment.data.sequence=null`). This will either deterministically stream over all datasets, or by default randomly shuffle classes. For a new dataset to be include, please ensure that you provide a list of classnames following the examples into `stream_construction/datasets`.

In both cases, you can include `experiment.evaluation.additional_datasets=[list_of_dataset_names]` to define datasets you __only__ want to evaluate over. Note that some datasets like `imagenet_a` __only__ have test/eval splits, so can't be streamed over.

By default, the preferred way (as it is the more powerful and controlled option)

___Creating a custom experiment config___

Once you have defined how you want to stream, you can create a new `.yaml`-configuration in `configs/experiment`. The way configurations work in this pipeline is:

* By default, we populate all arguments following the default configuration in `configs/config.yaml`. These argument have lowest priority.
* Any argument set in a config file inside `configs/experiment` and called via `experiment=experiment.yaml` will overwrite the default arguments. This means that you experimental config should also have the same overall structure.
* Finally, commandline argument have highest priority and will overwrite anything else, i.e. `python main.py experiment=experiment.yaml continual.method=finetune` will overwrite the `continual.method` flag in both `configs/config.yaml` and `configs/experiments/experiment.yaml`!

### 3.2 Determining Compute Budgets: Profile & Count GFlops

To profile each method's exact GFLOP utilisation, run:

```bash
python flops_computation.py experiment=simple_finetune.yaml continual.method=<method> experiment.backbone.name=openclip_vit_b32 experiment.dataset.name=['cifar100'] experiment.task.batch_size=1 log.name=flops_computation
```

### 3.3 Fomo-in-Flux training: Key parameters

___Selecting streaming models and datasets___

___Essential optimizer parameters___

By default, the following optimizer setup is utilized:

* We use an Adam-W optimizer, i.e. `experiment.optimizer.name=adamw` (which has been shown to work well with training large models),
* no label smoothing
* weight decay set as `experiment.optimizer.weight_decay=0.2` following e.g. the OpenCLIP setup,
* gradient clipping to `experiment.optimizer.clip_grad_norm=1`.

Moreover, we don't conduct batch-size based learning rate scaling by default (`experiment.optimizer.scaled_learning_rate=False`), i.e. via

$$ lr = \text{experiment.optimizer.lr} * \frac{\text{experiment.task.batch-size}}{256}$$

### 3.4 Running on data tar files

Particularly when deploying on high performance clusters, there is often a limit on files that can be stored. To circumvent this issue, it is also possible to run __FiF__ on tarballs directly. For this, simply set `experiment.dataset.tar=True`.

### 3.5 Multi-GPU and RAM preloading

___GPU id & Number of GPUs___

To change the ID of the utilized GPU, simply set

```bash
gpu=[<id_integer>]
```

To use multiple (_n_) gpus, simply set

```bash
gpu=[<id_integer_1>, ..., <id_integer_n>]
```

Note that currently, only `DataParallel` is provided for multi-gpu usage, but will be upgraded to `DistributedDataParallel` in the future.

To change the number of CPU cores used, simply adjust

```bash
experiment.dataset.num_workers=<num_workers>
```

___Preload all relevant datasets to memory___

By default, the dataloader will load the required data from the disk, unless

```bash
experiment.dataset.preload=True
```

is set, which will first load all required datasets (both train & test) into memory.
__Note:__ For __FiF__, this generally requires access to more than 80GB of RAM for a job, but ideally even more than that to avoid `OOM` during longer runs..

Finally, since adaptation happens on data chunks in __FiF__, and only testing requires access to all datasets at once, one may also only load the required evaluation/test data to memory:

```bash
experiment.dataset.preload_test_only=True
```

### 3.6 Adding your own twist to FiF

There will certainly come a time where you wish to move away from the default FiF setup. For this, we detail the three most common scenarios. For any other question, please simply raise an issue!.

#### 3.6.1 Custom dataset

Including a new dataset is ___fairly___ simple, but has to follow these steps:

__[1]__ First, create a respective dataloader in `data_lib`. For this, simply follow example structures e.g. in `data_lib/food101.py`. In particular, you need to define __five__ entries:

1. `BASE_CLASSES`, which defines the list of base classnames. In this pipeline, we provide these lists in `data_lib/00_info/<dataset_classnames.json>`.
2. `PRIMER`, which defines how _by default_ (changed when actual captions are used) classnames should be primed.
3. `CLASSES`, which generally just defines the primed base classnames.
4. `PARAMS`, which links to all the aforementioned properties, but also ideally for completeness (not needed for continual pretraining as we normalize based on pretraining statistics) dataset mean and standard deviation, default image size, `create_resized_variant_if_possible` referring to resizing of dataset images to `img_size`, whether it is an evaluation-only dataset (`eval_only`), and whether it is a classification or retrieval dataset (`type=<classification_or_retrieval`).
5. `class Dataset` inheriting from `data_lib.DatasetScaffold`, defining a `__init__` which provides a `self.root` to the dataset folder, `self.PARAMS` and calls `self._run_default_setup()`. For most datasets, we also include `_download_and_setup()` which is called at the very beginning for each dataset, downloading and setting up respective datasets. This is not needed / won't be called if the data was downloaded following the provided preprocessed links.

__[2]__ In `data_lib/00_info`, provide a `dataset.yaml` file which for each image path contains information on captions and more (compare e.g. `Food101_train_224.yaml`). For all provided datasets, these `.yaml`-files also include the 2-stage generated in-depth captions used for FIF training.

__[3]__ In `stream_construction/datasets`, include a `dataset.json` with a list of all classnames using the stream formatting.

#### 3.6.2 Custom continual learner

A new continual pretraining method simply inherits from `continual_lib.BaseContinualLearner`, and defines the following key functionalities:

1. `__init__()`: Importantly, defines `self.to_optimize`, the list of parameters to optimize for. Everything else is method-specific.
2. `observe()`: This function is called at every streaming iteration. Given input batch arguments, it performs an `autocasted` forward step, followed by a call of `self.gradient_update(loss)` on the computed loss.
3. (optional) `checkpoint`: A propery, returning everything to be checkpointed. Dictionary.
4. (optional) `load_from_checkpoint()`: How to take the checkpoint dictionary and initialize itself.
5. (optional) `end_task`: What to do at the end of every task. E.g. for merging methods, this generally means storing the current task weights into `self.checkpoint_storage`.
6. (optional) `begin_task`: What to do before each task training begins. Generally initializes the optimizer and scheduler for a given task.
7. (optional) `prepare_for_training` and `prepare_for_evaluation`: Generally refer to updates on current base model weights to use as a starting point for training and evaluation, respectively.

These functions are generally called in `utils/training.py`, which contains the __main training loop__, and `utils/evaluation.py`, which contains all evaluation functionalities.

#### 3.6.3 Custom backbone

If you want to include a new backbone, ensure that it is listed in `backbones.__init__.py > model_dict`, where the key references the backbone calling handle for `backbone.name`, and then contains a list of the form

```python
[embedding_dim, name_of_classification_head_if_provided, [list_of_accessible_image_backbone_blocks s.a. model.layer3], [list_of_accessible_text_backbone_blocks s.a. model.text_layer3], patch_embedding_handle]
```

Finally, in the same `__init__.py`, define in `get_backbone()` how this backbone is to be initialized and set up. Voila, that's all!

### 3.7 Running on validation splits

By default, each method will be evaluated periodically on respective test data - either that assigned to Fomo-in-Flux or the respective benchmarks. To perform e.g. hyperparameter optimization on validation data, please turn on

```bash
experiment.dataset.validation_mode=True
```

which will use the randomized train-validation-split defined in

```bash
experiment.dataset.train_val_split=0.8
```

### 3.8 Improving evaluation speeds during streaming

To efficiently evaluate generalization performance on a list of tasks after each training task, we also provide the option to only do so over a subset of the evaluation/test/validation data, going over the full sets only before (for zeroshot performance) and after adaptation (for final performance). This can be done simply via

```bash
evaluation.validate_on_subset=<Number in (0,1) denoting the percentage per dataset used.>
```

Note that the percentage per dataset used is generally capped at a minimum of 300 samples and at least two samples per class.

### 3.9 Evaluation from W&B xid

We provide a straightforward script to retrieve the exact knowledge accumulation and zero-shot retention metrics from a WandB run. We recommend running all experiments with WandB as our codebase heavily uses it for logging and evaluation. Given a wandb run, evaluation can be done using:

```bash
python results_from_wandb_xid.py --xid <wandb-experiment-run-id> --project <wandb-project> --entity <wandb-entity>
```

This script will return the averaged (across tasks) knowledge accumulation score, the zero-shot retention score, and the geometric mean.

### 3.10 W&B logging and checkpointing

___Weights & Biases Logging___

If one wishes to log results (requires online-logging using `Weights & Biases`), first set up an account at `wandb.ai`. Then, simply run any run with

```bash
log.use=True
```

To change the utilized `Weights & Biases` key, simply set it using

```bash
log.wandb_key=<your_key>
```

By default, logging (to `Weights & Biases`) is turned __on__ (`=True`).

___Checkpointing___

By default, checkpointing is turned on, which will store updated model weights and additional hyperparameter context in a parent folder defined in

```bash
log.folder=./checkpoints
```

In this folder, the script will search for a particular run name assigned to a particular run folder, associated with either `log.name`, or if not set, using the following rule:

```python
run_name = f'{log.group}_s-{experiment.seed}'
log_folder = os.path.join(
    log.folder, log.project, experiment.type, experiment.name, continual.method, run_name)
```

Note that `log.project` and `log.group` are changeable independently of each respective run for more precise checkpointing and placement if `log.name` is not set.

To turn off checkpointing, simply append

```bash
log.checkpoint=False
```
