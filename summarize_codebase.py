import importlib
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pathlib import PosixPath
from PIL import Image
import termcolor
import torchvision
import tqdm

import data_lib

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(args: DictConfig) -> None:
    np.random.seed(0)

    args.experiment.dataset.name = []
    args.experiment.evaluation.additional_datasets = []

    exclude = ['retinopathy', 'medmnist_derma', 'medmnist_organc', 'medmnist_organs']
    for name, handle in data_lib.DATASETS.items():
        if name not in exclude:
            mod = importlib.import_module(handle)
            if mod.PARAMS['eval_only']:
                args.experiment.evaluation.additional_datasets.append(name)
            else:
                args.experiment.dataset.name.append(name)

    print('\n')
    termcolor.cprint('Loading test/evaluation split for each available dataset!', color='yellow', attrs=['bold'])

    datasets_dict = data_lib.get_datasets(
        args, train_transform=args.experiment.dataset.train_transforms, test_transform=args.experiment.dataset.test_transforms)

    ### Checking that each file required by each dataset actually exists.
    print('\n')
    total_count = np.sum([len(datasets_dict[split]) for split in ['train', 'test', 'eval_only_test']])
    with tqdm.tqdm(total=total_count) as pbar:
        for split in ['train', 'test', 'eval_only_test']:
            for dataset in datasets_dict[split]:
                root = dataset.root.split('/')[-1].lower()
                num_files = len(dataset.data)
                pbar.set_description_str(f'Testing Completeness for {root} (split {split}, {num_files} files)...')
                for file in dataset.data:
                    if not os.path.exists(file):
                        raise Exception(f'File {file} in dataset {root} (split {split}) does not exist!')
                pbar.update(1)
    termcolor.cprint('All folders correctly set up!', color='green', attrs=['bold'])

    ### Printing overall summary.
    total_num_adapt_classes = np.sum([dataset.PARAMS['num_classes'] for dataset in datasets_dict['train'] if dataset.PARAMS['num_classes'] is not None])
    total_num_adapt_samples = np.sum([len(dataset) for dataset in datasets_dict['train']])
    total_num_eval_only_classes = np.sum([dataset.PARAMS['num_classes'] for dataset in datasets_dict['eval_only_test'] if dataset.PARAMS['num_classes'] is not None])
    total_num_eval_only_samples = np.sum([len(dataset) for dataset in datasets_dict['eval_only_test']])
    num_adapt_datasets = len(datasets_dict['train'])
    num_eval_only_datasets = len(datasets_dict['eval_only_test'])

    print('\n')
    termcolor.cprint('Data Summary', color='yellow', attrs=['bold'])
    termcolor.cprint('Adaption-Capable Datasets', color='white', attrs=['bold'])
    print(f'  - Number of Datasets: {num_adapt_datasets}')
    print(f'  - Number of Classes: {total_num_adapt_classes}')
    print(f'  - Number of Samples: {total_num_adapt_samples}')
    termcolor.cprint('Evaluation-Only Datasets', color='white', attrs=['bold'])
    print(f'  - Number of Datasets: {num_eval_only_datasets}')
    print(f'  - Number of Classes: {total_num_eval_only_classes}')
    print(f'  - Number of Samples: {total_num_eval_only_samples}')


    ### Creating sample visualization.
    print('\n')
    termcolor.cprint('Creating sample visualizations for each dataset!', color='yellow', attrs=['bold'])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation = torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None)
    ])

    save_folder = 'dataset_visualizations'
    os.makedirs(save_folder, exist_ok=True)
    for split in ['train', 'test', 'eval_only_test']:
        print(f'Generating sample visualizations for split [{split}]...')
        for dataset in tqdm.tqdm(datasets_dict[split], desc=f'Parsing datasets'):
            root = dataset.root.split('/')[-1].lower()
            num_samples = len(dataset)
            subset = np.random.choice(num_samples, 20, replace=False)
            samples, targets, captions, syn_captions, classes = [], [], [], [], []
            try:
                for i in subset:
                    samples.append(dataset.data[i])
                    targets.append(dataset.targets[i])
                    caption_data = dataset.caption_data
                    if caption_data is not None:
                        caption = caption_data[i]
                        if isinstance(caption, list):
                            caption = caption[0]
                        captions.append(caption)
                    else:
                        captions.append('N/A')
                    classnames = dataset.PARAMS['classes']
                    if classnames is not None:
                        classes.append(classnames[dataset.targets[i]])
                    else:
                        classes.append('N/A')
            except:
                from IPython import embed; embed()

            if isinstance(samples[0], str) or isinstance(samples[0], PosixPath):
                for i in range(len(samples)):
                    samples[i] = np.array(transform(Image.open(samples[i])))
            else:
                for i in range(len(samples)):
                    samples[i] = np.array(transform(Image.fromarray(samples[i])))

            f, axes = plt.subplots(4, 5)
            for i, ax in enumerate(axes.reshape(-1)):
                ax.imshow(samples[i])
                class_text = classes[i]
                caption = captions[i]
                title = ''
                txt_len = 30
                numchars = len(class_text)
                num_lines = int(np.ceil(numchars/txt_len))
                for k in range(num_lines):
                    title += class_text[k*txt_len:(k+1)*txt_len]
                    if k < num_lines-1:
                        title += '\n'
                caption_txt = ''
                numchars = len(caption)
                num_lines = int(np.ceil(numchars/txt_len))
                for k in range(num_lines):
                    caption_txt += caption[k*txt_len:(k+1)*txt_len]
                    if k < num_lines-1:
                        caption_txt += '\n'
                title = f'Cls: {title} [{targets[i]}]'
                title += f'\nCap: {caption_txt}'
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
            f.set_size_inches(5 * 4, 4 * 6)
            f.tight_layout()
            f.savefig(os.path.join(save_folder, f'{root}_{split}.jpg'))
            plt.close()

if __name__ == '__main__':
    main()