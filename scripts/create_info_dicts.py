#%%
import json
import os
from typing import Tuple

import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image
import torch
import torchvision.datasets
import torchvision.transforms
import tqdm

import data_lib

#%% Base paths
infd_path = './data_lib/00_info'

#%% ##########################################################    AI2DIAGRAMS
import data_lib.ai2diagrams
splits = ['train', 'test']
root = './data/AI2DIAGRAMS'
data_url = 'http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip'
# torchvision.datasets.utils.download_and_extract_archive(data_url, download_root=root)

cats = json.load(open(f'{root}/ai2d/categories.json', 'r'))
classnames = np.array([cats[key] for key in list(cats.keys())])
classnames_to_use = {
    'atomStructure': 'atom structure diagram',
    'circuits': 'circuits diagram',
    'eclipses': 'eclipses diagram',
    'faultsEarthquakes': 'diagram of earthquakes and faults',
    'foodChainsWebs': 'food chains and food webs diagram',
    'lifeCycles': 'life cycles diagram',
    'moonPhaseEquinox': 'diagram of moon phases and equinox',
    'partsOfTheEarth': 'diagram showing parts of the earth',
    'photosynthesisRespiration': 'diagram of photosynthesis and respiration',
    'rockCycle': 'diagram of rock cycles',
    'rockStrata': 'diagram of rock strata',
    'solarSystem': 'diagram of the solar system',
    'typesOf': 'diagram of sub-types or different forms of something',
    'volcano': 'volcano diagram',
    'waterCNPCycle': 'water / carbon / resource cycle diagram' 
}

image_paths = np.array([f'ai2d/images/{x}' for x in list(cats.keys())])
idcs = np.array([i for i, x in enumerate(classnames) if x in classnames_to_use])
cls2tar = {key: i for i, key in enumerate(classnames_to_use.keys())}
classnames = classnames[idcs]
image_paths = image_paths[idcs]
targets = np.array([cls2tar[cn] for cn in classnames])
test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))

ordered_class_names = [classnames_to_use[key] for key in classnames_to_use.keys()]

AI2DIAGRAMS_infodicts = {}
for split in splits:
    AI2DIAGRAMS_infodicts[split] = {}
    idcs = train_idcs if split == 'train' else test_idcs
    data_list = image_paths[idcs]
    target_list = targets[idcs]
    classname_list = classnames[idcs]
    
    for path, target, classname in zip(data_list, target_list, classname_list):
        classname = ordered_class_names[target]
        AI2DIAGRAMS_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.ai2diagrams.PRIMER.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(AI2DIAGRAMS_infodicts['train'], open(f'{infd_path}/AI2DIAGRAMS_train.json', 'w'), indent=4)
json.dump(AI2DIAGRAMS_infodicts['test'], open(f'{infd_path}/AI2DIAGRAMS_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/AI2DIAGRAMS_classnames.json', 'w'), indent=4)


#%% ##########################################################    ARTBENCH10
import os
import data_lib.artbench10
root = './data/ARTBENCH10'
# data_url = 'https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar'
# torchvision.datasets.utils.download_and_extract_archive(data_url, download_root=root)
# metadata_url = 'https://artbench.eecs.berkeley.edu/files/ArtBench-10.csv'
# os.system(f'wget -O {root}/metadata.csv {metadata_url}')

metadata = pd.read_csv(f'{root}/metadata.csv')

classnames = list(metadata['artist'])
a, b = np.unique(classnames, return_counts=True)
classes_to_remove = a[b < 5]
ordered_class_names = sorted([x for x in np.unique(classnames) if x not in classes_to_remove])
ordered_class_names = [x.replace('-', ' ') for x in ordered_class_names]

cls2tar = {clsn: i for i, clsn in enumerate(ordered_class_names)}
idcs = [i for i, y in tqdm.tqdm(enumerate(metadata['artist']), total=len(metadata['artist'])) if y not in classes_to_remove]

classnames = np.array(metadata['artist'])[idcs]
classnames = np.array([x.replace('-', ' ') for x in classnames])
image_paths = np.array(metadata['name'])[idcs]
targets = np.array([cls2tar[x] for x in classnames])
labels = np.array(metadata['label'])[idcs]
splits = np.array(metadata['split'])[idcs]
files = np.array([f'artbench-10-imagefolder-split/{split}/{label}/{image_path}' for split, label, image_path in zip(splits, labels, image_paths)])

test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))


ARTBENCH10_infodicts = {}
for split in ['train', 'test']:
    ARTBENCH10_infodicts[split] = {}
    idcs = train_idcs if split == 'train' else test_idcs
    data_list = files[idcs]
    target_list = targets[idcs]
    classname_list = classnames[idcs]
    
    
    for path, target, classname in zip(data_list, target_list, classname_list):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ARTBENCH10_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.artbench10.PRIMER.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(ARTBENCH10_infodicts['train'], open(f'{infd_path}/ARTBENCH10_train.json', 'w'), indent=4)
json.dump(ARTBENCH10_infodicts['test'], open(f'{infd_path}/ARTBENCH10_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/ARTBENCH10_classnames.json', 'w'), indent=4)



#%% ##########################################################    BIRDSNAP
import data_lib.birdsnap
root = './data/BIRDSNAP'
splits = ['train','test']

classes = sorted(os.listdir(f'{root}/birdsnap/download/images'))
ordered_class_names = [x.replace('_', ' ') for x in classes]

image_paths = []
targets = []
for i, classname in enumerate(classes):
    files = sorted(os.listdir(f'{root}/birdsnap/download/images/{classname}'))
    for file in files:
        image_paths.append(f'birdsnap/download/images/{classname}/{file}')
        targets.append(i)
del image_paths[23182]
del targets[23182]
image_paths = np.array(image_paths)
targets = np.array(targets)

test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))

BIRDSNAP_infodicts = {}
ordered_class_names = [x.replace('_', ' ') for x in ordered_class_names]
for split in splits:
    BIRDSNAP_infodicts[split] = {}
    idcs = train_idcs if split == 'train' else test_idcs
    data_list = image_paths[idcs]
    target_list = targets[idcs]
    
    for path, target in zip(data_list, target_list):
        classname = ordered_class_names[target]
        BIRDSNAP_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.birdsnap.PRIMER.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(BIRDSNAP_infodicts['train'], open(f'{infd_path}/BIRDSNAP_train.json', 'w'), indent=4)
json.dump(BIRDSNAP_infodicts['test'], open(f'{infd_path}/BIRDSNAP_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/BIRDSNAP_classnames.json', 'w'), indent=4)



#%% ##########################################################    CALTECH101
import data_lib.caltech101
root = './data/CALTECH101'
splits = ['train', 'test']
caltech101_dataset = torchvision.datasets.Caltech101(root, transform=None, target_transform=None, download=True)
caption_dict = pkl.load(open('./data/dataset_captions/CALTECH101_captions.pkl', 'rb'))

train_test_split = 0.7        
targets = np.array(caltech101_dataset.y)
data = []
for i in range(len(targets)):
    r1 = caltech101_dataset.categories[targets[i]]
    r2 = f"image_{caltech101_dataset.index[i]:04d}.jpg"
    data.append(os.path.join(root, 'caltech101', '101_ObjectCategories', r1, r2))

data_dict = {}
for target, path in zip(targets, data):
    if target not in data_dict:
        data_dict[target] = []
    data_dict[target].append(path)

train_data_dict = {key: values[:int(len(values)*train_test_split)] for key, values in data_dict.items()}
test_data_dict = {key: values[int(len(values)*train_test_split):] for key, values in data_dict.items()}

caltech101_infodicts = {}
ordered_class_names = [x.replace('_', ' ') for x in caltech101_dataset.categories]
splits = ['train', 'test']
for split, data_dict in zip(splits, [train_data_dict, test_data_dict]):
    caltech101_infodicts[split] = {}
    for target, items in data_dict.items():
        for path in items:
            classname = ordered_class_names[target]
            classinfo = '/'.join(path.split('/')[-2:])
            ref_path = path.replace(f'{root}/','')
            ref_path = f'CALTECH101/CALTECH101_{split}_224/{classinfo}'
            caltech101_infodicts[split][path.replace(f'{root}/','')] = {
                'classname': classname,
                'default_caption': None,
                'primer_caption': data_lib.caltech101.PRIMER.format(classname),
                'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
                'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
                'target': int(target)
            }

json.dump(caltech101_infodicts['train'], open(f'{infd_path}/CALTECH101_train.json', 'w'), indent=4)
json.dump(caltech101_infodicts['test'], open(f'{infd_path}/CALTECH101_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/CALTECH101_classnames.json', 'w'), indent=4)





#%% ##########################################################    CALTECH256
import data_lib.caltech256
root = './data/CALTECH256'
splits = ['train', 'test']
caltech256_dataset = torchvision.datasets.Caltech256(root, transform=None, target_transform=None, download=True)
caption_dict = pkl.load(open('./data/dataset_captions/CALTECH256_captions.pkl', 'rb'))

train_test_split = 0.7        
targets = np.array(caltech256_dataset.y)
data = []
for i in range(len(targets)):
    r1 = caltech256_dataset.categories[targets[i]]
    r2 = f"{caltech256_dataset.y[i] + 1:03d}_{caltech256_dataset.index[i]:04d}.jpg"
    data.append(os.path.join(root, 'caltech256', '256_ObjectCategories', r1, r2))

data_dict = {}
for target, path in zip(targets, data):
    if target not in data_dict:
        data_dict[target] = []
    data_dict[target].append(path)

train_data_dict = {key: values[:int(len(values)*train_test_split)] for key, values in data_dict.items()}
test_data_dict = {key: values[int(len(values)*train_test_split):] for key, values in data_dict.items()}

caltech256_infodicts = {}
ordered_class_names = [x.replace('_', ' ') for x in caltech256_dataset.categories]
splits = ['train', 'test']
for split, data_dict in zip(splits, [train_data_dict, test_data_dict]):
    caltech256_infodicts[split] = {}
    for target, items in data_dict.items():
        for path in items:
            classname = ordered_class_names[target]
            classinfo = '/'.join(path.split('/')[-2:])
            ref_path = path.replace(f'{root}/','')
            ref_path = f'CALTECH256/CALTECH256_{split}_224/{classinfo}'
            caltech256_infodicts[split][path.replace(f'{root}/','')] = {
                'classname': '.'.join(classname.split('.')[1:]).replace('-', ' ').replace('_', ' ').replace(' 101', ''),
                'default_caption': None,
                'primer_caption': data_lib.caltech256.PRIMER.format(classname),
                'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
                'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
                'target': int(target)
            }
ordered_class_names = ['.'.join(x.split('.')[1:]).replace('-', ' ').replace('_', ' ').replace(' 101', '') for x in caltech256_dataset.categories]
json.dump(caltech256_infodicts['train'], open(f'{infd_path}/CALTECH256_train.json', 'w'), indent=4)
json.dump(caltech256_infodicts['test'], open(f'{infd_path}/CALTECH256_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/CALTECH256_classnames.json', 'w'), indent=4)


#%% ##########################################################    CIFAR100
import data_lib.cifar100
root = './data/CIFAR100'
splits = ['train', 'test']
cifar100_datasets = {
    'train': torchvision.datasets.CIFAR100(root, True, transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.CIFAR100(root, False, transform=None, target_transform=None, download=True)
}
cifar100_caption_dict = pkl.load(open('./data/dataset_captions/CIFAR100_captions.pkl', 'rb'))

ordered_class_names = cifar100_datasets['train'].classes
cifar100_infodicts = {}

for split in splits:
    if split == 'train':
        data = list(range(len(cifar100_datasets['train'].data)))
    else:
        data = list(range(len(cifar100_datasets['train'].data), len(cifar100_datasets['train'].data) + len(cifar100_datasets['test'].data)))
    targets = cifar100_datasets[split].targets
    conversion = {val:key for key, val in cifar100_datasets[split].class_to_idx.items()}
    classnames = []
    for target in targets:
        classnames.append(conversion[target])

    if split == 'train':
        ordered_class_names = [conversion[x] for x in sorted(np.unique(targets))]
        
    cifar100_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        sub = 0 if split == 'train' else len(cifar100_datasets['train'].data)
        key_path = f'cifar-100-images-{split}/{path-sub}.png'        
        cifar100_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.cifar100.PRIMER.format(classname),
            'synthetic_caption': cifar100_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': cifar100_caption_dict[ref_path]['merged_caption'],
            'target': target
        }
    cifar100_infodicts[split] = cifar100_infodict

json.dump(cifar100_infodicts['train'], open(f'{infd_path}/CIFAR100_train.json', 'w'), indent=4)
json.dump(cifar100_infodicts['test'], open(f'{infd_path}/CIFAR100_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/CIFAR100_classnames.json', 'w'), indent=4)





#%% ##########################################################    CIFAR10
import data_lib.cifar10
root = './data/CIFAR10'
splits = ['train', 'test']
cifar10_datasets = {
    'train': torchvision.datasets.CIFAR10(root, True, transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.CIFAR10(root, False, transform=None, target_transform=None, download=True)
}
cifar10_caption_dict = pkl.load(open('./data/dataset_captions/CIFAR10_captions.pkl', 'rb'))

ordered_class_names = cifar10_datasets['train'].classes
cifar10_infodicts = {}

for split in splits:
    if split == 'train':
        data = list(range(len(cifar10_datasets['train'].data)))
    else:
        data = list(range(len(cifar10_datasets['train'].data), len(cifar10_datasets['train'].data) + len(cifar10_datasets['test'].data)))
    targets = cifar10_datasets[split].targets
    conversion = {val:key for key, val in cifar10_datasets[split].class_to_idx.items()}
    classnames = []
    for target in targets:
        classnames.append(conversion[target])

    if split == 'train':
        ordered_class_names = [conversion[x] for x in sorted(np.unique(targets))]
        
    cifar10_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        sub = 0 if split == 'train' else len(cifar10_datasets['train'].data)
        key_path = f'cifar-10-images-{split}/{path-sub}.png'        
        cifar10_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.cifar10.PRIMER.format(classname),
            'synthetic_caption': cifar10_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': cifar10_caption_dict[ref_path]['merged_caption'],
            'target': target
        }
    cifar10_infodicts[split] = cifar10_infodict

json.dump(cifar10_infodicts['train'], open(f'{infd_path}/CIFAR10_train.json', 'w'), indent=4)
json.dump(cifar10_infodicts['test'], open(f'{infd_path}/CIFAR10_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/CIFAR10_classnames.json', 'w'), indent=4)





#%% ##########################################################    CARS196
import data_lib.cars196
root = './data/CARS196'
splits = ['train', 'test']
cars196_datasets = {
    'train':torchvision.datasets.StanfordCars(root, split='train', transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.StanfordCars(root, split='test', transform=None, target_transform=None, download=True)
}
caption_dict = pkl.load(open('./data/dataset_captions/CARS196_captions.pkl', 'rb'))

cars196_infodicts = {}
for split in splits:
    data = [('./' + x[0]).replace(root + '/', '') for x in cars196_datasets[split]._samples]
    targets = [x[1] for x in cars196_datasets[split]._samples]
    conversion = {val:key for key, val in cars196_datasets[split].class_to_idx.items()}
    classnames = []
    for target in targets:
        classnames.append(conversion[target])
    path_conv = {
        path: f"CARS196/CARS196_{split}_224/{classname.replace('/', '-')}/{path.split('/')[-1]}" for path, classname in zip(data, classnames)
    }
    rev_path_conv = {val: key for key, val in path_conv.items()}

    if split == 'train':
        ordered_class_names = [conversion[x] for x in sorted(np.unique(targets))]
        
    cars196_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path_conv[path]
        cars196_infodict[path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.cars196.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': target
        }
        
    cars196_infodicts[split] = cars196_infodict

json.dump(cars196_infodicts['train'], open(f'{infd_path}/CARS196_train.json', 'w'), indent=4)
json.dump(cars196_infodicts['test'], open(f'{infd_path}/CARS196_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/CARS196_classnames.json', 'w'), indent=4)






#%% ##########################################################    COUNTRY211
import data_lib.country211
root = './data/COUNTRY211'
splits = ['train', 'test']
country211_datasets = {
    'train':torchvision.datasets.Country211(root, split='train', transform=None, target_transform=None, download=False),
    'test': torchvision.datasets.Country211(root, split='test', transform=None, target_transform=None, download=False)
}
country211_caption_dict = pkl.load(open('./data/dataset_captions/COUNTRY211_captions.pkl', 'rb'))

country211_classes = [
    'Andorra', 'United Arab Emirates', 'Afghanistan', 'Antigua and Barbuda', 'Anguilla', 'Albania', 'Armenia', 'Angola',
    'Antarctica', 'Argentina', 'Austria', 'Australia', 'Aruba', 'Aland Islands', 'Azerbaijan', 'Bosnia and Herzegovina',
    'Barbados', 'Bangladesh', 'Belgium', 'Burkina Faso', 'Bulgaria', 'Bahrain', 'Benin', 'Bermuda', 'Brunei Darussalam',
    'Bolivia', 'Bonaire, Saint Eustatius and Saba', 'Brazil', 'Bahamas', 'Bhutan', 'Botswana', 'Belarus', 'Belize',
    'Canada', 'DR Congo', 'Central African Republic', 'Switzerland', "Cote d'Ivoire", 'Cook Islands', 'Chile',
    'Cameroon', 'China', 'Colombia', 'Costa Rica', 'Cuba', 'Cabo Verde', 'Curacao', 'Cyprus', 'Czech Republic',
    'Germany', 'Denmark', 'Dominica', 'Dominican Republic', 'Algeria', 'Ecuador', 'Estonia', 'Egypt', 'Spain',
    'Ethiopia', 'Finland', 'Fiji', 'Falkland Islands', 'Faeroe Islands', 'France', 'Gabon', 'United Kingdom', 'Grenada',
    'Georgia', 'French Guiana', 'Guernsey', 'Ghana', 'Gibraltar', 'Greenland', 'Gambia', 'Guadeloupe', 'Greece',
    'South Georgia and South Sandwich Is.', 'Guatemala', 'Guam', 'Guyana', 'Hong Kong', 'Honduras', 'Croatia', 'Haiti',
    'Hungary', 'Indonesia', 'Ireland', 'Israel', 'Isle of Man', 'India', 'Iraq', 'Iran', 'Iceland', 'Italy', 'Jersey',
    'Jamaica', 'Jordan', 'Japan', 'Kenya', 'Kyrgyz Republic', 'Cambodia', 'St. Kitts and Nevis', 'North Korea',
    'South Korea', 'Kuwait', 'Cayman Islands', 'Kazakhstan', 'Laos', 'Lebanon', 'St. Lucia', 'Liechtenstein',
    'Sri Lanka', 'Liberia', 'Lithuania', 'Luxembourg', 'Latvia', 'Libya', 'Morocco', 'Monaco', 'Moldova', 'Montenegro',
    'Saint-Martin', 'Madagascar', 'Macedonia', 'Mali', 'Myanmar', 'Mongolia', 'Macau', 'Martinique', 'Mauritania',
    'Malta', 'Mauritius', 'Maldives', 'Malawi', 'Mexico', 'Malaysia', 'Mozambique', 'Namibia', 'New Caledonia',
    'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal', 'New Zealand', 'Oman', 'Panama', 'Peru',
    'French Polynesia', 'Papua New Guinea', 'Philippines', 'Pakistan', 'Poland', 'Puerto Rico', 'Palestine', 'Portugal',
    'Palau', 'Paraguay', 'Qatar', 'Reunion', 'Romania', 'Serbia', 'Russia', 'Rwanda', 'Saudi Arabia', 'Solomon Islands',
    'Seychelles', 'Sudan', 'Sweden', 'Singapore', 'St. Helena', 'Slovenia', 'Svalbard and Jan Mayen Islands',
    'Slovakia', 'Sierra Leone', 'San Marino', 'Senegal', 'Somalia', 'South Sudan', 'El Salvador', 'Sint Maarten',
    'Syria', 'Eswatini', 'Togo', 'Thailand', 'Tajikistan', 'Timor-Leste', 'Turkmenistan', 'Tunisia', 'Tonga', 'Turkey',
    'Trinidad and Tobago', 'Taiwan', 'Tanzania', 'Ukraine', 'Uganda', 'United States', 'Uruguay', 'Uzbekistan',
    'Vatican', 'Venezuela', 'British Virgin Islands', 'United States Virgin Islands', 'Vietnam', 'Vanuatu', 'Samoa',
    'Kosovo', 'Yemen', 'South Africa', 'Zambia', 'Zimbabwe',
]

country211_infodicts = {}
for split in splits:
    data = [('./' + x[0]).replace(root + '/', '') for x in country211_datasets[split].imgs]
    targets = [x[1] for x in country211_datasets[split].imgs]
    handles = []
    classnames = []
    for target in targets:
        classnames.append(country211_classes[target])
        handles.append(country211_datasets[split].classes[target])
    path_conv = {
        path: f"COUNTRY211/COUNTRY211_{split}_224/{handle}/{path.split('/')[-1]}" for path, handle in zip(data, handles)
    }
    rev_path_conv = {val: key for key, val in path_conv.items()}

    country211_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path_conv[path]
        country211_infodict[path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.country211.PRIMER.format(classname),
            'synthetic_caption': country211_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': country211_caption_dict[ref_path]['merged_caption'],
            'target': target
        }
        
    country211_infodicts[split] = country211_infodict

json.dump(country211_infodicts['train'], open(f'{infd_path}/COUNTRY211_train.json', 'w'), indent=4)
json.dump(country211_infodicts['test'], open(f'{infd_path}/COUNTRY211_test.json', 'w'), indent=4)
json.dump(country211_classes, open(f'{infd_path}/COUNTRY211_classnames.json', 'w'), indent=4)








#%% ##########################################################    CUB200
import data_lib.cub200
root = './data/CUB200'
splits = ['train', 'test']
cub200_datasets = {
    'train': np.load(f'{root}/train_data.npz', allow_pickle=True),
    'test': np.load(f'{root}/test_data.npz', allow_pickle=True)
}
cub200_caption_dict = pkl.load(open('./data/dataset_captions/CUB200_captions.pkl', 'rb'))

ordered_class_names = [x[1].replace('_', ' ').replace('-', ' ') for x in cub200_datasets['train']['classes']]

cub200_infodicts = {}
train_len = len(cub200_datasets['train']['data'])
test_len = len(cub200_datasets['test']['data'])

for split in splits:
    if split == 'train':
        data = list(range(train_len))
    else:
        data = list(range(train_len, train_len + test_len))    
    targets = cub200_datasets[split]['targets']
    classnames = []
    for target in targets:
        classnames.append(ordered_class_names[target])

    cub200_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        sub = 0 if split == 'train' else train_len
        key_path = f'cub-200-images-{split}/{path-sub}.png'                
        cub200_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.cub200.PRIMER.format(classname),
            'synthetic_caption': cub200_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': cub200_caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }
    cub200_infodicts[split] = cub200_infodict

json.dump(cub200_infodicts['train'], open(f'{infd_path}/CUB200_train.json', 'w'), indent=4)
json.dump(cub200_infodicts['test'], open(f'{infd_path}/CUB200_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/CUB200_classnames.json', 'w'), indent=4)


#%% ##########################################################    DF20MINI
import data_lib.df20mini
import pandas as pd

root = './data/DF20MINI'
splits = ['train', 'test']
df20mini_data = {
    'train': pd.read_csv(f'{root}/DF20M-train_metadata_PROD.csv'),
    'test': pd.read_csv(f'{root}/DF20M-public_test_metadata_PROD.csv')
}
caption_dict = pkl.load(open('./data/dataset_captions/DF20MINI_captions.pkl', 'rb'))
for split in splits:
    df20mini_data[split] = df20mini_data[split].dropna(subset=['species'])

ordered_class_names = sorted(os.listdir(os.path.join(root, 'train')))

data_dict = {split: [] for split in splits}
data_classes_dict = {split: [] for split in splits}
targets_dict = {split: [] for split in splits}

for split in splits:
    for i, folder in enumerate(ordered_class_names):
        for file in os.listdir(os.path.join(root, split, folder)):
            data_dict[split].append(os.path.join(split, folder, file))
            targets_dict[split].append(i)
            data_classes_dict[split].append(folder)

df20mini_infodicts = {}
for split in splits:
    data = data_dict[split]
    targets = targets_dict[split]
    data_classes = data_classes_dict[split]
    
    df20mini_infodicts[split] = {}
    
    for path, target, data_class in zip(data, targets, data_classes):        
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'DF20MINI/DF20MINI_{split}_224/{classinfo}'
        df20mini_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.df20mini.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(df20mini_infodicts['train'], open(f'{infd_path}/DF20MINI_train.json', 'w'), indent=4)
json.dump(df20mini_infodicts['test'], open(f'{infd_path}/DF20MINI_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DF20MINI_classnames.json', 'w'), indent=4)




#%% ##########################################################    DOLLARSTREET
import data_lib.dollarstreet
import pandas as pd

root = './data/DOLLARSTREET'
splits = ['train', 'test']

general_data = pd.read_csv(f'{root}/dataset_dollarstreet/images_v2.csv')
train_metadata = pd.read_csv(f'{root}/dataset_dollarstreet/images_v2_imagenet_train.csv')
test_metadata = pd.read_csv(f'{root}/dataset_dollarstreet/images_v2_imagenet_test.csv')
total_metadata = pd.concat([train_metadata, test_metadata])


data = {}
for split in splits:
    metadata = train_metadata if split == 'train' else test_metadata
    data[split] = {}
    data[split]['image_paths'] = np.array(metadata['imageRelPath'])
    data[split]['country_names'] = np.array(metadata['country.name'])
    data[split]['topics'] = np.array(metadata['topics'])
    data[split]['places'] = np.array(metadata['place'])
    data[split]['income'] = np.array(metadata['income'])
    data[split]['in_classes'] = np.array(metadata['imagenet_synonyms'])

train_classes = [f'{eval(x)[0]} from {y}' for x, y in zip(data['train']['in_classes'], data['train']['country_names'])]
test_classes = [f'{eval(x)[0]} from {y}' for x, y in zip(data['test']['in_classes'], data['test']['country_names'])]

ordered_class_names = sorted(list(set(np.unique(train_classes)).intersection(set(np.unique(test_classes)))))
cls2tar = {classname: i for i, classname in enumerate(ordered_class_names)}
train_idcs = np.array([i for i, x in tqdm.tqdm(enumerate(train_classes), total=len(train_classes)) if x in  ordered_class_names])
test_idcs = np.array([i for i, x in tqdm.tqdm(enumerate(test_classes), total=len(test_classes)) if x in  ordered_class_names])
for split in splits:
    idcs = train_idcs if split == 'train' else test_idcs
    classes = train_classes if split == 'train' else test_classes
    for key in data[split].keys():
        data[split][key] = data[split][key][idcs]
    data[split]['classnames'] = np.array(classes)[idcs]
    data[split]['targets'] = [cls2tar[classname] for classname in data[split]['classnames']]
     
DOLLARSTREET_infodicts = {}
for split in splits:
    paths = data[split]['image_paths']
    targets = data[split]['targets']
    
    DOLLARSTREET_infodicts[split] = {}
    
    for path, target in zip(paths, targets):        
        classname = ordered_class_names[target]
        DOLLARSTREET_infodicts[split]['dataset_dollarstreet/'+path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.dollarstreet.PRIMER.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(DOLLARSTREET_infodicts['train'], open(f'{infd_path}/DOLLARSTREET_train.json', 'w'), indent=4)
json.dump(DOLLARSTREET_infodicts['test'], open(f'{infd_path}/DOLLARSTREET_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DOLLARSTREET_classnames.json', 'w'), indent=4)


#%% ##########################################################    DOMAINNET_CLIPART
import data_lib.domainnet_clipart
import pandas as pd

root = './data/DOMAINNET_CLIPART'
splits = ['train', 'test']
dnclip_data = {
    'train': pd.read_csv(os.path.join(root, f'clipart_train.txt'), delimiter=' ', header=None),
    'test': pd.read_csv(os.path.join(root, f'clipart_test.txt'), delimiter=' ', header=None)
}
caption_dict = pkl.load(open('./data/dataset_captions/DOMAINNET_CLIPART_captions.pkl', 'rb'))

train_paths = [x for x in dnclip_data['train'][0]]
test_paths = [x for x in dnclip_data['test'][0]]
train_targets = [x for x in dnclip_data['train'][1]]
test_targets = [x for x in dnclip_data['test'][1]]
train_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]
test_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]

target_to_classname = {}
for class_name, target in zip(train_default_class_names, train_targets):
    target_to_classname[target] = class_name

ordered_class_names = [target_to_classname[i] for i in range(len(target_to_classname))]

classname_dict = {'train': train_default_class_names, 'test': test_default_class_names}
path_dict = {'train': train_paths, 'test': test_paths}
target_dict = {'train': train_targets, 'test': test_targets}

dnclip_infodicts = {}
for split in splits:
    data = path_dict[split]
    targets = target_dict[split]
    classnames = classname_dict[split]
    
    dnclip_infodicts[split] = {}
    
    for path, target in zip(data, targets):        
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'DOMAINNET_CLIPART/DOMAINNET_CLIPART_{split}_224/{classinfo}'
        dnclip_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.domainnet_clipart.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(dnclip_infodicts['train'], open(f'{infd_path}/DOMAINNET_CLIPART_train.json', 'w'), indent=4)
json.dump(dnclip_infodicts['test'], open(f'{infd_path}/DOMAINNET_CLIPART_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DOMAINNET_CLIPART_classnames.json', 'w'), indent=4)





#%% ##########################################################    DOMAINNET_INFOGRAPH
import data_lib.domainnet_infograph
import pandas as pd

root = './data/DOMAINNET_INFOGRAPH'
splits = ['train', 'test']
dn_data = {
    'train': pd.read_csv(os.path.join(root, f'infograph_train.txt'), delimiter=' ', header=None),
    'test': pd.read_csv(os.path.join(root, f'infograph_test.txt'), delimiter=' ', header=None)
}
caption_dict = pkl.load(open('./data/dataset_captions/DOMAINNET_INFOGRAPH_captions.pkl', 'rb'))

train_paths = [x for x in dn_data['train'][0]]
test_paths = [x for x in dn_data['test'][0]]
train_targets = [x for x in dn_data['train'][1]]
test_targets = [x for x in dn_data['test'][1]]
train_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]
test_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]

target_to_classname = {}
for class_name, target in zip(train_default_class_names, train_targets):
    target_to_classname[target] = class_name

ordered_class_names = [target_to_classname[i] for i in range(len(target_to_classname))]

classname_dict = {'train': train_default_class_names, 'test': test_default_class_names}
path_dict = {'train': train_paths, 'test': test_paths}
target_dict = {'train': train_targets, 'test': test_targets}

dn_infodicts = {}
for split in splits:
    data = path_dict[split]
    targets = target_dict[split]
    classnames = classname_dict[split]
    
    dn_infodicts[split] = {}
    
    for path, target in zip(data, targets):        
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'DOMAINNET_INFOGRAPH/DOMAINNET_INFOGRAPH_{split}_224/{classinfo}'
        dn_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.domainnet_infograph.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(dn_infodicts['train'], open(f'{infd_path}/DOMAINNET_INFOGRAPH_train.json', 'w'), indent=4)
json.dump(dn_infodicts['test'], open(f'{infd_path}/DOMAINNET_INFOGRAPH_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DOMAINNET_INFOGRAPH_classnames.json', 'w'), indent=4)





#%% ##########################################################    DOMAINNET_PAINTING
import data_lib.domainnet_painting
import pandas as pd

root = './data/DOMAINNET_PAINTING'
splits = ['train', 'test']
dn_data = {
    'train': pd.read_csv(os.path.join(root, f'painting_train.txt'), delimiter=' ', header=None),
    'test': pd.read_csv(os.path.join(root, f'painting_test.txt'), delimiter=' ', header=None)
}
caption_dict = pkl.load(open('./data/dataset_captions/DOMAINNET_PAINTING_captions.pkl', 'rb'))

train_paths = [x for x in dn_data['train'][0]]
test_paths = [x for x in dn_data['test'][0]]
# paintings misses class 327.
train_targets = [x  if x < 327 else x-1 for x in dn_data['train'][1]]
test_targets = [x  if x < 327 else x-1 for x in dn_data['test'][1]]
train_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]
test_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]

target_to_classname = {}
for class_name, target in zip(train_default_class_names, train_targets):
    target_to_classname[target] = class_name

ordered_class_names = [target_to_classname[i] for i in range(len(target_to_classname))]

classname_dict = {'train': train_default_class_names, 'test': test_default_class_names}
path_dict = {'train': train_paths, 'test': test_paths}
target_dict = {'train': train_targets, 'test': test_targets}

dn_infodicts = {}
for split in splits:
    data = path_dict[split]
    targets = target_dict[split]
    classnames = classname_dict[split]
    
    dn_infodicts[split] = {}
    
    for path, target in zip(data, targets):        
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'DOMAINNET_PAINTING/DOMAINNET_PAINTING_{split}_224/{classinfo}'
        dn_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.domainnet_painting.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(dn_infodicts['train'], open(f'{infd_path}/DOMAINNET_PAINTING_train.json', 'w'), indent=4)
json.dump(dn_infodicts['test'], open(f'{infd_path}/DOMAINNET_PAINTING_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DOMAINNET_PAINTING_classnames.json', 'w'), indent=4)











#%% ##########################################################    DOMAINNET_QUICKDRAW
import data_lib.domainnet_quickdraw
import pandas as pd

root = './data/DOMAINNET_QUICKDRAW'
splits = ['train', 'test']
dn_data = {
    'train': pd.read_csv(os.path.join(root, f'quickdraw_train.txt'), delimiter=' ', header=None),
    'test': pd.read_csv(os.path.join(root, f'quickdraw_test.txt'), delimiter=' ', header=None)
}
caption_dict = pkl.load(open('./data/dataset_captions/DOMAINNET_QUICKDRAW_captions.pkl', 'rb'))

train_paths = [x for x in dn_data['train'][0]]
test_paths = [x for x in dn_data['test'][0]]
train_targets = [x for x in dn_data['train'][1]]
test_targets = [x for x in dn_data['test'][1]]

train_classnames = {}
train_datadict = {}
for target, data in zip(train_targets, train_paths):
    train_classnames[target] = data.split('/')[-2].replace('_', ' ')
    if target not in train_datadict:
        train_datadict[target] = []
    train_datadict[target].append(data)
subset = 0.5
train_datadict = {key: sorted(values)[:int(len(values) * subset)] for key, values in train_datadict.items()}
cleaned_train_paths = list(train_datadict.values())
cleaned_train_paths = [x for y in cleaned_train_paths for x in y]

test_classnames = {}
test_datadict = {}
for target, data in zip(test_targets, test_paths):
    test_classnames[target] = data.split('/')[-2].replace('_', ' ')
    if target not in test_datadict:
        test_datadict[target] = []
    test_datadict[target].append(data)
subset = 0.5
test_datadict = {key: sorted(values)[:int(len(values) * subset)] for key, values in test_datadict.items()}
cleaned_test_paths = list(test_datadict.values())
cleaned_test_paths = [x for y in cleaned_test_paths for x in y]

train_idcs = [i for i,x in tqdm.tqdm(enumerate(train_paths), total=len(train_paths)) if x in cleaned_train_paths]
test_idcs = [i for i,x in tqdm.tqdm(enumerate(test_paths), total=len(test_paths)) if x in cleaned_test_paths]

train_paths = [x for i, x in tqdm.tqdm(enumerate(train_paths), total=len(train_paths)) if i in train_idcs]
test_paths = [x for i, x in tqdm.tqdm(enumerate(test_paths), total=len(test_paths)) if i in test_idcs]
train_targets = [x for i, x in tqdm.tqdm(enumerate(train_targets), total=len(train_paths)) if i in train_idcs]
test_targets = [x for i, x in tqdm.tqdm(enumerate(test_targets), total=len(test_paths)) if i in test_idcs]

train_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]
test_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]

target_to_classname = {}
for class_name, target in zip(train_default_class_names, train_targets):
    target_to_classname[target] = class_name

ordered_class_names = [target_to_classname[i] for i in range(len(target_to_classname))]

classname_dict = {'train': train_default_class_names, 'test': test_default_class_names}
path_dict = {'train': train_paths, 'test': test_paths}
target_dict = {'train': train_targets, 'test': test_targets}

dn_infodicts = {}
for split in splits:
    data = path_dict[split]
    targets = target_dict[split]
    classnames = classname_dict[split]
    
    dn_infodicts[split] = {}
    
    for path, target in zip(data, targets):        
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'DOMAINNET_QUICKDRAW/DOMAINNET_QUICKDRAW_{split}_224/{classinfo}'
        dn_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.domainnet_quickdraw.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(dn_infodicts['train'], open(f'{infd_path}/DOMAINNET_QUICKDRAW_train.json', 'w'), indent=4)
json.dump(dn_infodicts['test'], open(f'{infd_path}/DOMAINNET_QUICKDRAW_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DOMAINNET_QUICKDRAW_classnames.json', 'w'), indent=4)






#%% ##########################################################    DOMAINNET_SKETCH
import data_lib.domainnet_sketch
import pandas as pd

root = './data/DOMAINNET_SKETCH'
splits = ['train', 'test']
dn_data = {
    'train': pd.read_csv(os.path.join(root, f'sketch_train.txt'), delimiter=' ', header=None),
    'test': pd.read_csv(os.path.join(root, f'sketch_test.txt'), delimiter=' ', header=None)
}
caption_dict = pkl.load(open('./data/dataset_captions/DOMAINNET_SKETCH_captions.pkl', 'rb'))

train_paths = [x for x in dn_data['train'][0]]
test_paths = [x for x in dn_data['test'][0]]
train_targets = [x for x in dn_data['train'][1]]
test_targets = [x for x in dn_data['test'][1]]
train_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]
test_default_class_names = [x.split('/')[1].replace('_', ' ').replace('-', ' ') for x in train_paths]

target_to_classname = {}
for class_name, target in zip(train_default_class_names, train_targets):
    target_to_classname[target] = class_name

ordered_class_names = [target_to_classname[i] for i in range(len(target_to_classname))]

classname_dict = {'train': train_default_class_names, 'test': test_default_class_names}
path_dict = {'train': train_paths, 'test': test_paths}
target_dict = {'train': train_targets, 'test': test_targets}

dn_infodicts = {}
for split in splits:
    data = path_dict[split]
    targets = target_dict[split]
    classnames = classname_dict[split]
    
    dn_infodicts[split] = {}
    
    for path, target in zip(data, targets):        
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'DOMAINNET_SKETCH/DOMAINNET_SKETCH_{split}_224/{classinfo}'
        dn_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.domainnet_sketch.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(dn_infodicts['train'], open(f'{infd_path}/DOMAINNET_SKETCH_train.json', 'w'), indent=4)
json.dump(dn_infodicts['test'], open(f'{infd_path}/DOMAINNET_SKETCH_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DOMAINNET_SKETCH_classnames.json', 'w'), indent=4)








#%% ##########################################################    DSPRITES
import data_lib.dsprites
import pandas as pd
import itertools as it

root = './data/DSPRITES'
splits = ['train', 'test']

ordered_class_names = [
    "square located in the top left", "square located on the center left side", "square located in the bottom left", "square located in the upper central area", "square located in the center area", "square located in the bottom central area", "square located in the top right", "square located on the central right side", "square located in the bottom right", 
    "ellipse located in the top left", "ellipse located on the center left side", "ellipse located in the bottom left", "ellipse located in the upper central area", "ellipse located in the center area", "ellipse located in the bottom central area", "ellipse located in the top right", "ellipse located on the central right side", "ellipse located in the bottom right", 
    "heart located in the top left", "heart located on the center left side", "heart located in the bottom left", "heart located in the upper central area", "heart located in the center area", "heart located in the bottom central area", "heart located in the top right", "heart located on the central right side", "heart located in the bottom right"
]

compressed_data = np.load(f'{root}/dsprites.npz')
imgs = compressed_data['imgs']
latent_values = compressed_data['latents_values']

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
avail_latents = np.array(list(it.product(range(1), range(3), range(6), range(40), range(27), range(27))))
train_indices = list(np.random.choice(len(avail_latents), 75000, replace=False))
test_indices = list(np.random.choice(list(set(range(len(avail_latents))) - set(train_indices)), 25000, replace=False))
train_latents = avail_latents[sorted(train_indices)]
test_latents = avail_latents[sorted(test_indices)]
mul = np.array([[524880, 174960, 29160, 729, 27, 1]])
train_indices = np.sum(train_latents * mul, axis=-1)
test_indices = np.sum(test_latents * mul, axis=-1)

#bottom, center, top
#left, center, right
ll = []            
for lab in tqdm.tqdm(sub_latents, desc='Generating label data...'):
    x = int(lab[-2] > 0.33) + int(lab[-2] > 0.66)
    y = int(lab[-1] > 0.33) + int(lab[-1] > 0.66)
    s = int(lab[1])
    label = (s-1) * 9 + x * 3 + y
    ll.append(label)
ll = np.array(ll).astype(int)

lat_names = ('color', 'shape', 'scale', 'orientation', 'posX', 'posY')
lat_sizes = np.array([1, 3, 6, 40, 32, 32])

ors = sub_latents[:, 3]
xs = sub_latents[:, -2]
ys = sub_latents[:, -1]

full_phrases = [
    "{0} positioned at x = {1} (0 means left, 1 right) and y = {2} (0 signifies top, 1 bottom). The {3}'s orientation is given by {4}, from 0 to 2*pi.",
    "{0} (exact position: x = {1} (0: left, 1: right) and y = {2} (0: top, 1: bottom). Orientation is {4} (0 to 2*pi)",
    "{0}: Located at x = {1} (with 0 as left, 1 as right) and y = {2} (0 top, 1 bottom). The angle of orientation for the {3} is {4} (0 - 2*pi)",
    "{0}: Precise coordinates are x = {1} (0: left side, 1: right side) and y = {2} (0 the top, 1 the bottom). The directional orientation is {4} (0 to 2*pi)",
    "{0}. Position on the x-axis is {1} (0 to 1, left to right) and {2} (0 to 1, top to bottom) on the y-axis. The orientation of the {3} is {4} (varies from 0 to 2*pi)"    
]    
short_phrases = [
    "{0} positioned at x = {1} (0 means left, 1 right) and y = {2} (0 signifies top, 1 bottom)",
    "{0} (exact position: x = {1} (0: left, 1: right) and y = {2} (0: top, 1: bottom)",
    "{0}: Located at x = {1} (with 0 as left, 1 as right) and y = {2} (0 top, 1 bottom)",
    "{0}: Precise coordinates are x = {1} (0: left side, 1: right side) and y = {2} (0 the top, 1 the bottom)",
    "{0}. Position on the x-axis is {1} (0 to 1, left to right) and {2} (0 to 1, top to bottom) on the y-axis"    
]

captions = []
np.random.seed(0)
for o, x, y, label in tqdm.tqdm(zip(ors, xs, ys, ll), total=len(ors)):
    classname = ordered_class_names[label]
    shape = classname.split(' ')[0]
    x = np.round(x, 3)
    y = np.round(y, 3)
    o = np.round(o, 3)
    val = np.random.choice(11)
    if val == 0:
        caption = classname
    if val in range(1,6):        
        caption = np.random.choice(full_phrases).format(classname, x, y, shape, o)
    if val in range(6,11):        
        caption = np.random.choice(short_phrases).format(classname, x, y)
    caption = data_lib.dsprites.PRIMER.format(caption)
    captions.append(caption) 
captions = np.array(captions)


train_data = sub_imgs[train_indices]
train_targets = ll[train_indices]
test_data = sub_imgs[test_indices]
test_targets = ll[test_indices]
train_captions = captions[train_indices]
test_captions = captions[test_indices]

all_data = {'train': train_indices, 'test': test_indices}
all_targets = {'train': train_targets, 'test': test_targets}
all_captions = {'train': train_captions, 'test': test_captions}

dsprites_infodicts = {}
for split in splits:
    sub_data = all_data[split]
    sub_targets = all_targets[split]
    sub_captions = all_captions[split]
    
    dsprites_infodicts[split] = {}
    
    for i, (idx, target, caption) in enumerate(zip(sub_data, sub_targets, sub_captions)):
        classname = ordered_class_names[target]
        path = f'dsprites-images-{split}/{idx}.png'
        dsprites_infodicts[split][path] = {
            'classname': classname,
            'default_caption': caption,
            'primer_caption': data_lib.dsprites.PRIMER.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(dsprites_infodicts['train'], open(f'{infd_path}/DSPRITES_train.json', 'w'), indent=4)
json.dump(dsprites_infodicts['test'], open(f'{infd_path}/DSPRITES_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DSPRITES_classnames.json', 'w'), indent=4)

#%% ##########################################################    DTD
import data_lib.dtd
root = './data/DTD'
splits = ['train', 'test']
dtd_datasets = {
    'train': torchvision.datasets.DTD(root, split='train', transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.DTD(root, split='test', transform=None, target_transform=None, download=True)
}
caption_dict = pkl.load(open('./data/dataset_captions/DTD_captions.pkl', 'rb'))

data_dict = {split: [str(x).replace('data/DTD/','') for x in dtd_datasets[split]._image_files] for split in splits}
target_dict = {split: dtd_datasets[split]._labels for split in splits}

dtd_infodicts = {}
ordered_class_names = [x.replace('_', ' ') for x in dtd_datasets['train'].classes]
for split in splits:
    dtd_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'DTD/{path}'
        dtd_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.dtd.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(dtd_infodicts['train'], open(f'{infd_path}/DTD_train.json', 'w'), indent=4)
json.dump(dtd_infodicts['test'], open(f'{infd_path}/DTD_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/DTD_classnames.json', 'w'), indent=4)






#%% ##########################################################    EUROSAT
import data_lib.eurosat
root = './data/EuroSAT'
splits = ['train', 'test']
eurosat_dataset = torchvision.datasets.EuroSAT(root, transform=None, target_transform=None, download=True)
caption_dict = pkl.load(open('./data/dataset_captions/EuroSAT_captions.pkl', 'rb'))
data = [x[0] for x in eurosat_dataset.samples]
targets = np.array([x[1] for x in eurosat_dataset.samples])

train_test_split = 0.7
num_samples_per_classes = {key: np.where(targets == key)[0] for key in np.unique(targets)}
train_idcs = {key: list(item[:int(train_test_split * len(item))]) for key, item in num_samples_per_classes.items()}
test_idcs = {key: list(item[int(train_test_split * len(item)):]) for key, item in num_samples_per_classes.items()}
train_idcs = sorted([x for y in train_idcs.values() for x in y])
test_idcs = sorted([x for y in test_idcs.values() for x in y])

data_dict = {
    'train': [x for i, x in tqdm.tqdm(enumerate(data), total=len(data)) if i in train_idcs],
    'test': [x for i, x in tqdm.tqdm(enumerate(data), total=len(data)) if i in test_idcs]
}
target_dict = {'train': targets[train_idcs], 'test': targets[test_idcs]}

eurosat_infodicts = {}
ordered_class_names = [
    "annual crop land",
    "forest",
    "brushland or shrubland",
    "highway or road",
    "industrial buildings or commercial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings or homes or apartments",
    "river",
    "lake or sea"
]

for split in splits:
    eurosat_infodicts[split] = {}
    paths = data_dict[split]
    paths = [x.replace('./data/EuroSAT/', '') for x in paths]
    targets = target_dict[split]
    for path, target in zip(paths, targets):
        classname = ordered_class_names[target]
        ref_path = f'EuroSAT/{path}'
        eurosat_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.eurosat.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(eurosat_infodicts['train'], open(f'{infd_path}/EuroSAT_train.json', 'w'), indent=4)
json.dump(eurosat_infodicts['test'], open(f'{infd_path}/EuroSAT_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/EuroSAT_classnames.json', 'w'), indent=4)






#%% ##########################################################    FashionMNIST
import data_lib.fashionmnist
root = './data/FashionMNIST'
splits = ['train', 'test']
datasets = {
    'train': torchvision.datasets.FashionMNIST(root, True, transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.FashionMNIST(root, False, transform=None, target_transform=None, download=True)
}

fashionmnist_caption_dict = pkl.load(open('./data/dataset_captions/FashionMNIST_captions.pkl', 'rb'))
ordered_class_names = list(datasets['train'].classes)
fashionmnist_infodicts = {}

for split in splits:
    data = list(range(len(datasets[split].data)))
    targets = np.array(datasets[split].targets)
    conversion = {val:key for key, val in datasets[split].class_to_idx.items()}
    classnames = []
    for target in targets:
        classnames.append(conversion[target])

    fashionmnist_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        key_path = f'fashionmnist-images-{split}/{target}-{path}.png'        
        fashionmnist_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.fashionmnist.PRIMER.format(classname),
            'synthetic_caption': fashionmnist_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': fashionmnist_caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }
    fashionmnist_infodicts[split] = fashionmnist_infodict

json.dump(fashionmnist_infodicts['train'], open(f'{infd_path}/FashionMNIST_train.json', 'w'), indent=4)
json.dump(fashionmnist_infodicts['test'], open(f'{infd_path}/FashionMNIST_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/FashionMNIST_classnames.json', 'w'), indent=4)


#%% ##########################################################    FGVCAircraft
import data_lib.fgvcaircraft
root = './data/FGVCAircraft'
splits = ['train', 'test']
FGVCAircraft_datasets = {
    'train': torchvision.datasets.FGVCAircraft(root, split='train', transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.FGVCAircraft(root, split='test', transform=None, target_transform=None, download=True)
}
caption_dict = pkl.load(open('./data/dataset_captions/FGVCAircraft_captions.pkl', 'rb'))

data_dict = {split: [str(x).replace('data/FGVCAircraft/','') for x in FGVCAircraft_datasets[split]._image_files] for split in splits}
target_dict = {split: FGVCAircraft_datasets[split]._labels for split in splits}

FGVCAircraft_infodicts = {}
ordered_class_names = [x.replace('_', ' ') for x in FGVCAircraft_datasets['train'].classes]
for split in splits:
    FGVCAircraft_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        path = path[2:]
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'FGVCAircraft/{path}'
        FGVCAircraft_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.fgvcaircraft.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(FGVCAircraft_infodicts['train'], open(f'{infd_path}/FGVCAircraft_train.json', 'w'), indent=4)
json.dump(FGVCAircraft_infodicts['test'], open(f'{infd_path}/FGVCAircraft_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/FGVCAircraft_classnames.json', 'w'), indent=4)


#%% ##########################################################    FLICKR30K
import data_lib.flickr30k
import pandas as pd

root = './data/FLICKR30K'
metadata = pd.read_csv(f'{root}/annotations.csv')                
data = [f'{root}/flickr30k-images/{file}' for file, split in zip(metadata.filename, metadata.split) if split == 'test']

caption_data = [eval(x) for x, split in zip(metadata.raw, metadata.split) if split == 'test']
# For retrieval datasets, targets are simply the associated caption idcs.
targets = torch.arange(len(data)).to(torch.long)

flickr_infodict = {}
for path, caption, target in zip(data, caption_data, targets):        
    flickr_infodict[path.replace('./data/FLICKR30K/','')] = {
        'classname': None,
        'default_caption': caption,
        'primer_caption': None,
        'synthetic_caption': None,
        'synthetic_merged_caption': None,
        'target': int(target)
    }

json.dump(flickr_infodict, open(f'{infd_path}/FLICKR30K_test.json', 'w'), indent=4)



#%% ##########################################################    FLOWERS102
import data_lib.flowers102
root = './data/FLOWERS102'
splits = ['train', 'test']
FLOWERS102_datasets = {
    'train': torchvision.datasets.Flowers102(root, split='test', transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.Flowers102(root, split='train', transform=None, target_transform=None, download=True)
}
caption_dict = pkl.load(open('./data/dataset_captions/FLOWERS102_captions.pkl', 'rb'))

data_dict = {split: [str(x).replace('data/FLOWERS102/','') for x in FLOWERS102_datasets[split]._image_files] for split in splits}
target_dict = {split: FLOWERS102_datasets[split]._labels for split in splits}

FLOWERS102_infodicts = {}

ordered_class_names = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 
    'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 
    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 
    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 
    'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 
    'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 
    'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 
    'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 
    'thorn apple', 'morning glory', 'passion flower', 'lotus lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 
    'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 
    'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily'
]

for split in splits:
    FLOWERS102_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'FLOWERS102/{path}'
        FLOWERS102_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.flowers102.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(FLOWERS102_infodicts['train'], open(f'{infd_path}/FLOWERS102_train.json', 'w'), indent=4)
json.dump(FLOWERS102_infodicts['test'], open(f'{infd_path}/FLOWERS102_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/FLOWERS102_classnames.json', 'w'), indent=4)

#%% ##########################################################    Food101
import data_lib.food101
root = './data/Food101'
splits = ['train', 'test']
Food101_datasets = {
    'train': torchvision.datasets.Food101(root, split='train', transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.Food101(root, split='test', transform=None, target_transform=None, download=True)
}
caption_dict = pkl.load(open('./data/dataset_captions/Food101_captions.pkl', 'rb'))

data_dict = {split: [str(x).replace('data/Food101/','') for x in Food101_datasets[split]._image_files] for split in splits}
target_dict = {split: Food101_datasets[split]._labels for split in splits}

Food101_infodicts = {}

ordered_class_names = [x.replace('_', ' ') for x in Food101_datasets['train'].classes]

for split in splits:
    Food101_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'Food101/{path}'
        Food101_infodicts[split][str(path.replace(f'{root}/',''))] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.food101.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(Food101_infodicts['train'], open(f'{infd_path}/Food101_train.json', 'w'), indent=4)
json.dump(Food101_infodicts['test'], open(f'{infd_path}/Food101_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/Food101_classnames.json', 'w'), indent=4)

#%% ##########################################################    FRU92
import shutil
import zipfile
import data_lib.fru92
root = './data/FRU92'
splits = ['train', 'test']
veg200_root = './data/VEG200'
base_path = os.path.join(root, 'processed', 'vegfru_list')
# if 'archive.zip' in os.listdir(root):
#     archive_file = os.path.join(root, 'archive.zip')
# else:
#     archive_file = os.path.join(veg200_root, 'archive.zip')
    
# with zipfile.ZipFile(archive_file,"r") as zip_ref:
#     zip_ref.extractall(os.path.join(root, 'processed'))

# fru92_veg_path = os.path.join(root, 'processed', 'veg200_images')
# shutil.move(fru92_veg_path, os.path.join(veg200_root, 'processed'))
# shutil.copytree(base_path, os.path.join(veg200_root, 'processed', 'vegfru_list'))
            
files = {
    'train': pd.read_csv(os.path.join(base_path, f'vegfru_test.txt'), delimiter=' ', header=None),
    'test': pd.read_csv(os.path.join(base_path, f'vegfru_train.txt'), delimiter=' ', header=None)
}

file_dict = {split: files[split][0] for split in splits}
target_dict = {split: files[split][1] for split in splits}
data_dict = {split: [] for split in splits}
classnames = {}
for split in splits:
    targets =[]
    for i in tqdm.trange(len(file_dict[split])):
        if 'fru92_images' in file_dict[split][i]:
            data_dict[split].append(os.path.join(root, 'processed', file_dict[split][i]))
            target = target_dict[split][i]
            targets.append(target)
            classnames[target] = file_dict[split][i].split('/')[1].replace('_', ' ')
    min_target = min(targets)
    targets = [x - min_target for x in targets]      
    target_dict[split] = targets

caption_dict = pkl.load(open('./data/dataset_captions/FRU92_captions.pkl', 'rb'))

unique_targets = sorted(np.unique(target_dict[split]))
ordered_class_names = [classnames[target + min_target] for target in unique_targets]

FRU92_infodicts = {}
for split in splits:
    FRU92_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        path = str(path.replace(f'{root}/',''))
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'FRU92/FRU92_{split}_224/{classinfo}'
        FRU92_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.fru92.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(FRU92_infodicts['train'], open(f'{infd_path}/FRU92_train.json', 'w'), indent=4)
json.dump(FRU92_infodicts['test'], open(f'{infd_path}/FRU92_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/FRU92_classnames.json', 'w'), indent=4)


#%% ##########################################################    FSCOCO
root = './data/FSCOCO'
data_url = 'http://cvssp.org/data/fscoco/fscoco.tar.gz'
# torchvision.datasets.utils.download_and_extract_archive(data_url, download_root=root)

img_base_file = f'{root}/fscoco/raster_sketches'
text_base_file = f'{root}/fscoco/text'
trueimg_base_file = f'{root}/fscoco/images'
groups = sorted(os.listdir(img_base_file))

images = []
texts = []
trueimages = []
for group in tqdm.tqdm(groups):
    img_groupfile = f'{img_base_file}/{group}'
    text_groupfile = f'{text_base_file}/{group}'
    trueimg_groupfile = f'{trueimg_base_file}/{group}'
    
    img_files = [f'{img_groupfile}/{x}' for x in sorted(os.listdir(img_groupfile))]
    trueimg_files = [f'{trueimg_groupfile}/{x}' for x in sorted(os.listdir(trueimg_groupfile))]
    text_files = [list(pd.read_csv(f'{text_groupfile}/{x}'))[0] for x in sorted(os.listdir(text_groupfile))]
    
    trueimages.extend(trueimg_files)
    images.extend(img_files)
    texts.extend(text_files)
targets = np.arange(len(images))

import open_clip
backbone, _, _ = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s32b_b82k', cache_dir='./cache_dir')

tokenizer = open_clip.get_tokenizer('ViT-L-14')
import torchvision
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

import spacy
nlp = spacy.load("en_core_web_sm")

nsubjs = []
nouns_list = []
for i,text in tqdm.tqdm(enumerate(texts), total=len(texts)):
    lemma = None
    nn_root = None
    nouns = []
    for token in nlp(text):      
        if 'NN' in token.tag_ and len(token.lemma_) >= 3:
            nouns.append(token.lemma_.lower())
            
        if token.dep_ == 'nsubj':
            lemma = token.lemma_.lower()
        if token.tag_ == 'NN' and token.dep_ == 'ROOT':
            nn_root = token.lemma_.lower()
        if token.tag_ == 'NNP' and token.dep_ == 'ROOT':
            nn_root = token.lemma_.lower()
        if token.tag_ == 'NNS' and token.dep_ == 'ROOT':
            nn_root = token.lemma_.lower()
        if token.tag_ == 'NNPS' and token.dep_ == 'ROOT':
            nn_root = token.lemma_.lower()
    
    
    nouns_list.append(nouns)
    
    if lemma is not None or nn_root is not None:
        final_text = lemma if lemma is not None else nn_root
        nsubjs.append([i, final_text])

device = torch.device('cuda')
_ = backbone.to(device)

final_files = []
final_sketches = []
final_texts = []
final_class = []
for i in tqdm.tqdm(range(len(texts))):
    img_path = trueimages[i]
    sketch_path = images[i]
    
    nouns = [x.lower() for x in nouns_list[i]]
    if len(nouns) > 0:
        if len(nouns) == 1:
            final_class.append(nouns[0])
        else:
            with torch.amp.autocast("cuda"), torch.no_grad():
                img = transform(Image.open(img_path).convert('RGB')).to(device)
                img_embed = torch.nn.functional.normalize(backbone.encode_image(img.unsqueeze(0)), dim=-1)
                text_embed = torch.nn.functional.normalize(backbone.encode_text(tokenizer(['A photo of {}'.format(x) for x in nouns]).to(device)))
                ic = torch.argsort(img_embed @ text_embed.T, dim=1, descending=True)[0][:2].cpu().numpy()
                noun = np.array(nouns)[ic]
            final_class.append(noun[0])
        final_files.append(img_path)
        final_sketches.append(sketch_path)
        final_texts.append(texts[i])

final_files = np.array(final_files)
final_class = np.array(final_class)
final_sketches = np.array(final_sketches)
final_texts = np.array(final_texts)

a, b = np.unique(final_class, return_counts=True)
rem_classes = a[b < 10]
idcs = [i for i, x in enumerate(final_class) if x not in rem_classes]
final_files = final_files[idcs]
final_sketches = final_sketches[idcs]
final_texts = final_texts[idcs]
final_class = final_class[idcs]

final_class = [x.lower() for x in final_class]

merge = {
    ('aeroplane','aircraft','aircraft','areoplane'): 'airplane',
    ('disc', 'disk'): 'disc',
    ('giraffe','giraffes'): 'giraffe',
    ('sheep','sheeps'): 'sheep',
    ('zebra','zebras'): 'zebra',
    ('road','roadside'): 'road',
}

import copy
classnames = copy.deepcopy(final_class)
for i in range(len(classnames)):
    clsn = classnames[i]
    to_merge = [clsn in val for val in merge.keys()]
    if any(to_merge):
        clsn = merge[list(merge.keys())[int(np.where(to_merge)[0])]]
        classnames[i] = clsn
test_idcs = range(0, len(classnames), 5)
train_idcs = list(set(range(len(classnames))) - set(test_idcs))

final_files = np.array(final_files)
final_sketches = np.array(final_sketches)
final_texts = np.array(final_texts)
classnames = np.array(classnames)

ordered_class_names = sorted(np.unique(classnames))
cls2tar = {c: i for i, c in enumerate(ordered_class_names)}

fscoco_infodict = {'train': {}, 'test': {}}

for split in ['train', 'test']:
    idcs = train_idcs if split == 'train' else test_idcs
    file_list = final_sketches[idcs]
    text_list = final_texts[idcs]
    class_list = classnames[idcs]
    
    for path, caption, classname in zip(file_list, text_list, class_list):        
        target = cls2tar[classname]
        fscoco_infodict[split][path.replace('./data/FSCOCO/','')] = {
            'classname': classname,
            'default_caption': 'A sketch of {}'.format(caption),
            'primer_caption': None,
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(fscoco_infodict['train'], open(f'{infd_path}/FSCOCO_train.json', 'w'), indent=4)
json.dump(fscoco_infodict['test'], open(f'{infd_path}/FSCOCO_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/FSCOCO_classnames.json', 'w'), indent=4)


#%% ##########################################################    GTSRB
import data_lib.gtsrb
root = './data/GTSRB'
splits = ['train', 'test']

caption_dict = pkl.load(open('./data/dataset_captions/GTSRB_captions.pkl', 'rb'))

ordered_class_names = [
    'red and white circle 20 kph speed limit', 'red and white circle 30 kph speed limit', 'red and white circle 50 kph speed limit', 'red and white circle 60 kph speed limit', 
    'red and white circle 70 kph speed limit', 'red and white circle 80 kph speed limit', 'end / de-restriction of 80 kph speed limit', 'red and white circle 100 kph speed limit',
    'red and white circle 120 kph speed limit', 'red and white circle red car and black car no passing', 'red and white circle red truck and black car no passing', 'red and white triangle road intersection warning',
    'white and yellow diamond priority road', 'red and white upside down triangle yield right-of-way', 'stop', 'empty red and white circle',
    'red and white circle no truck entry', 'red circle with white horizonal stripe no entry', 'red and white triangle with exclamation mark warning', 'red and white triangle with black left curve approaching warning',
    'red and white triangle with black right curve approaching warning', 'red and white triangle with black double curve approaching warning', 'red and white triangle rough / bumpy road warning', 'red and white triangle car skidding / slipping warning',
    'red and white triangle with merging / narrow lanes warning', 'red and white triangle with person digging / construction / road work warning', 'red and white triangle with traffic light approaching warning', 'red and white triangle with person walking warning',
    'red and white triangle with child and person walking warning', 'red and white triangle with bicyle warning', 'red and white triangle with snowflake / ice warning', 'red and white triangle with deer warning',
    'white circle with gray strike bar no speed limit', 'blue circle with white right turn arrow mandatory', 'blue circle with white left turn arrow mandatory', 'blue circle with white forward arrow mandatory',
    'blue circle with white forward or right turn arrow mandatory', 'blue circle with white forward or left turn arrow mandatory', 'blue circle with white keep right arrow mandatory', 'blue circle with white keep left arrow mandatory',
    'blue circle with white arrows indicating a traffic circle', 'white circle with gray strike bar indicating no passing for cars has ended', 'white circle with gray strike bar indicating no passing for trucks has ended',
]

train_val_split = 0.7
data = [x[0] for x in torchvision.datasets.folder.make_dataset(root, extensions=(".ppm"))]
classes = [sample.split('GTSRB/Training/')[-1].split('/')[0] for sample in data]
data_dict = {}
for path, classname in zip(data, classes):
    if classname not in data_dict:
        data_dict[classname] = []
    data_dict[classname].append(path)

data_dicts = {
    'train': {key: item[:int(len(item) * train_val_split)] for key, item in data_dict.items()},
    'test': {key: item[int(len(item) * train_val_split):] for key, item in data_dict.items()}
}

class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(np.unique(classes)))}

data_dict = {split: [] for split in splits}
target_dict = {split: [] for split in splits}

for split in splits:
    for class_name, item in data_dicts[split].items():
        for path in item:
            data_dict[split].append(path)
            target_dict[split].append(class_to_idx[class_name])

GTSRB_infodicts = {}
for split in splits:
    GTSRB_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'GTSRB/GTSRB/Training/{classinfo}'
        GTSRB_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.gtsrb.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(GTSRB_infodicts['train'], open(f'{infd_path}/GTSRB_train.json', 'w'), indent=4)
json.dump(GTSRB_infodicts['test'], open(f'{infd_path}/GTSRB_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/GTSRB_classnames.json', 'w'), indent=4)

        

#%% ##########################################################    IMAGENET_A
import data_lib.imagenet_a
root = './data/IMAGENET_A'

caption_dict = pkl.load(open('./data/dataset_captions/IMAGENET_A_captions.pkl', 'rb'))

data = []
targets = []
folders = [x for x in sorted(os.listdir(os.path.join(root, 'imagenet-a'))) if x != 'README.txt']
for i, folder in enumerate(folders):
    for file in os.listdir(os.path.join(root, 'imagenet-a', folder)):
        data.append(os.path.join(root, 'imagenet-a', folder, file))
        targets.append(i)

ordered_class_names = [
    "stingray", "goldfinch", "junco", "American robin", "jay", "bald eagle", "vulture", "newt", "American bullfrog", "box turtle", "green iguana", "agama", "chameleon", "American alligator", "garter snake", "harvestman", "scorpion", "tarantula",
    "centipede", "sulphur-crested cockatoo", "lorikeet", "hummingbird", "toucan", "duck", "goose", "koala", "jellyfish", "sea anemone", "flatworm", "snail", "crayfish", "hermit crab", "flamingo", "great egret", "oystercatcher", "pelican", "sea lion", 
    "Chihuahua", "Golden Retriever", "Rottweiler", "German Shepherd Dog", "pug", "red fox", "Persian cat", "lynx", "lion", "American black bear", "mongoose", "ladybug", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "stick insect", "cockroach", "mantis", "leafhopper", "dragonfly", "monarch butterfly", "small white", "gossamer-winged butterfly", "starfish", "cottontail rabbit", 
    "porcupine", "fox squirrel", "marmot", "bison", "skunk", "armadillo", "baboon", "white-headed capuchin", "African bush elephant", "pufferfish", "academic gown", "accordion", "acoustic guitar", "airliner", "ambulance", "apron", "balance beam", "balloon", "banjo", "barn", "wheelbarrow", "basketball", "lighthouse", "beaker", "bikini",
    "bow", "bow tie", "breastplate", "broom", "candle", "canoe", "castle", "cello", "chain", "chest", "Christmas stocking", "cowboy boot", "cradle", "rotary dial telephone", "digital clock", "doormat", "drumstick", "dumbbell", "envelope", "feather boa", "flagpole", "forklift", "fountain", "garbage truck",
    "goblet", "go-kart", "golf cart", "grand piano", "hair dryer", "clothes iron", "jack-o'-lantern", "jeep", "kimono", "lighter", "limousine", "manhole cover", "maraca", "marimba", "mask", "mitten", "mosque", "nail", "obelisk", "ocarina", "organ", "parachute", "parking meter", "piggy bank", "billiard table", 
    "hockey puck", "quill", "racket", "reel", "revolver", "rocking chair", "rugby ball", "salt shaker", "sandal", "saxophone", "school bus", "schooner", "sewing machine", "shovel", "sleeping bag", "snowmobile", "snowplow", "soap dispenser", "spatula", "spider web", "steam locomotive", "stethoscope", "couch", "submarine", 
    "sundial", "suspension bridge", "syringe", "tank", "teddy bear", "toaster", "torch", "tricycle", "umbrella", "unicycle", "viaduct", "volleyball", "washing machine", "water tower", "wine bottle", "shipwreck", "guacamole", "pretzel", "cheeseburger", "hot dog", "broccoli", "cucumber", "bell pepper", "mushroom", "lemon", 
    "banana", "custard apple", "pomegranate", "carbonara", "bubble", "cliff", "volcano", "baseball player", "rapeseed", "yellow lady's slipper", "corn", "acorn"
]

IMAGENET_A_infodicts = {}
for path, target in zip(data, targets):
    classname = ordered_class_names[target]
    classinfo = '/'.join(path.split('/')[-2:])
    ref_path = f'IMAGENET_A/IMAGENET_A_test_224/{classinfo}'
    IMAGENET_A_infodicts[path.replace(f'{root}/','')] = {
        'classname': classname,
        'default_caption': None,
        'primer_caption': data_lib.imagenet_a.PRIMER.format(classname),
        'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
        'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
        'target': int(target)
    }

json.dump(IMAGENET_A_infodicts, open(f'{infd_path}/IMAGENET_A_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/IMAGENET_A_classnames.json', 'w'), indent=4)


#%% ##########################################################    IMAGENET
import data_lib.imagenet
root = './data/IMAGENET'

caption_dict = pkl.load(open('./data/dataset_captions/IMAGENET_captions.pkl', 'rb'))

data = []
targets = []
folders = [x for x in sorted(os.listdir(os.path.join(root, 'val'))) if x != 'README.txt']
for i, folder in enumerate(folders):
    for file in os.listdir(os.path.join(root, 'val', folder)):
        data.append(os.path.join(root, 'val', folder, file))
        targets.append(i)

ordered_class_names = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", 
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "eft", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", 
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", 
    "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", 
    "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", 
    "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", 
    "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", 
    "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", 
    "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", 
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", 
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", 
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", 
    "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", 
    "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", 
    "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", 
    "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", 
    "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", 
    "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", 
    "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", 
    "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", 
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", 
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", 
    "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", 
    "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", 
    "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", 
    "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", 
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", 
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", 
    "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", 
    "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", 
    "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", 
    "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "maillot", "one-piece bathing suit", "manhole cover", 
    "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", 
    "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", 
    "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", 
    "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", 
    "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", 
    "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", 
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", 
    "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", 
    "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", 
    "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", 
    "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", 
    "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", 
    "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", 
    "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", 
    "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", 
    "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", 
    "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", 
    "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
]

IMAGENET_infodicts = {}
for path, target in zip(data, targets):
    classname = ordered_class_names[target]
    classinfo = '/'.join(path.split('/')[-2:])
    ref_path = f'IMAGENET/IMAGENET_val_224/{classinfo}'
    IMAGENET_infodicts[path.replace(f'{root}/','')] = {
        'classname': classname,
        'default_caption': None,
        'primer_caption': data_lib.imagenet.PRIMER.format(classname),
        'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
        'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
        'target': int(target)
    }

json.dump(IMAGENET_infodicts, open(f'{infd_path}/IMAGENET_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/IMAGENET_classnames.json', 'w'), indent=4)


#%% ##########################################################    IMAGENET_R
import data_lib.imagenet_r
root = './data/IMAGENET_R'

caption_dict = pkl.load(open('./data/dataset_captions/IMAGENET_R_captions.pkl', 'rb'))

data = []
targets = []
folders = [x for x in sorted(os.listdir(os.path.join(root, 'imagenet-r'))) if x != 'README.txt']
for i, folder in enumerate(folders):
    for file in os.listdir(os.path.join(root, 'imagenet-r', folder)):
        data.append(os.path.join(root, 'imagenet-r', folder, file))
        targets.append(i)

ordered_class_names = [
    "goldfish", "great white shark", "hammerhead", "stingray", "hen", "ostrich", "goldfinch", "junco", "bald eagle", "vulture", "newt", "axolotl", "tree frog", 
    "iguana", "African chameleon", "cobra", "scorpion", "tarantula", "centipede", "peacock", "lorikeet", "hummingbird", "toucan", "duck", "goose", "black swan", 
    "koala", "jellyfish", "snail", "lobster", "hermit crab", "flamingo", "american egret", "pelican", "king penguin", "grey whale", "killer whale", "sea lion", "chihuahua", 
    "shih tzu", "afghan hound", "basset hound", "beagle", "bloodhound", "italian greyhound", "whippet", "weimaraner", "yorkshire terrier", "boston terrier", "scottish terrier", 
    "west highland white terrier", "golden retriever", "labrador retriever", "cocker spaniels", "collie", "border collie", "rottweiler", "german shepherd dog", "boxer", 
    "french bulldog", "saint bernard", "husky", "dalmatian", "pug", "pomeranian", "chow chow", "pembroke welsh corgi", "toy poodle", "standard poodle", "timber wolf", 
    "hyena", "red fox", "tabby cat", "leopard", "snow leopard", "lion", "tiger", "cheetah", "polar bear", "meerkat", "ladybug", "fly", "bee", "ant", "grasshopper", 
    "cockroach", "mantis", "dragonfly", "monarch butterfly", "starfish", "wood rabbit", "porcupine", "fox squirrel", "beaver", "guinea pig", "zebra", "pig", "hippopotamus", 
    "bison", "gazelle", "llama", "skunk", "badger", "orangutan", "gorilla", "chimpanzee", "gibbon", "baboon", "panda", "eel", "clown fish", "puffer fish", "accordion", 
    "ambulance", "assault rifle", "backpack", "barn", "wheelbarrow", "basketball", "bathtub", "lighthouse", "beer glass", "binoculars", "birdhouse", "bow tie", "broom", 
    "bucket", "cauldron", "candle", "cannon", "canoe", "carousel", "castle", "mobile phone", "cowboy hat", "electric guitar", "fire engine", "flute", "gasmask", "grand piano", 
    "guillotine", "hammer", "harmonica", "harp", "hatchet", "jeep", "joystick", "lab coat", "lawn mower", "lipstick", "mailbox", "missile", "mitten", "parachute", 
    "pickup truck", "pirate ship", "revolver", "rugby ball", "sandal", "saxophone", "school bus", "schooner", "shield", "soccer ball", "space shuttle", "spider web", 
    "steam locomotive", "scarf", "submarine", "tank", "tennis ball", "tractor", "trombone", "vase", "violin", "military aircraft", "wine bottle", "ice cream", "bagel", 
    "pretzel", "cheeseburger", "hotdog", "cabbage", "broccoli", "cucumber", "bell pepper", "mushroom", "Granny Smith", "strawberry", "lemon", "pineapple", "banana", 
    "pomegranate", "pizza", "burrito", "espresso", "volcano", "baseball player", "scuba diver", "acorn"
]

IMAGENET_R_infodicts = {}
for path, target in zip(data, targets):
    classname = ordered_class_names[target]
    classinfo = '/'.join(path.split('/')[-2:])
    ref_path = f'IMAGENET_R/IMAGENET_R_test_224/{classinfo}'
    IMAGENET_R_infodicts[path.replace(f'{root}/','')] = {
        'classname': classname,
        'default_caption': None,
        'primer_caption': data_lib.imagenet_r.PRIMER.format(classname),
        'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
        'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
        'target': int(target)
    }

json.dump(IMAGENET_R_infodicts, open(f'{infd_path}/IMAGENET_R_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/IMAGENET_R_classnames.json', 'w'), indent=4)


#%% ##########################################################    IMAGENET_S
import data_lib.imagenet_s
root = './data/IMAGENET_S'

caption_dict = pkl.load(open('./data/dataset_captions/IMAGENET_S_captions.pkl', 'rb'))

data = []
targets = []
folders = [x for x in sorted(os.listdir(os.path.join(root, 'sketch'))) if x != 'README.txt']
for i, folder in enumerate(folders):
    for file in os.listdir(os.path.join(root, 'sketch', folder)):
        data.append(os.path.join(root, 'sketch', folder, file))
        targets.append(i)

ordered_class_names = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", 
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "eft", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", 
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", 
    "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", 
    "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", 
    "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", 
    "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", 
    "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", 
    "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", 
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", 
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", 
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", 
    "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", 
    "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", 
    "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", 
    "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", 
    "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", 
    "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", 
    "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", 
    "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", 
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", 
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", 
    "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", 
    "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", 
    "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", 
    "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", 
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", 
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", 
    "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", 
    "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", 
    "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", 
    "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "maillot", "one-piece bathing suit", "manhole cover", 
    "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", 
    "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", 
    "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", 
    "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", 
    "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", 
    "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", 
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", 
    "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", 
    "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", 
    "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", 
    "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", 
    "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", 
    "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", 
    "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", 
    "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", 
    "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", 
    "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", 
    "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
]

IMAGENET_S_infodicts = {}
for path, target in zip(data, targets):
    classname = ordered_class_names[target]
    classinfo = '/'.join(path.split('/')[-2:])
    ref_path = f'IMAGENET_S/IMAGENET_S_test_224/{classinfo}'
    IMAGENET_S_infodicts[path.replace(f'{root}/','')] = {
        'classname': classname,
        'default_caption': None,
        'primer_caption': data_lib.imagenet_s.PRIMER.format(classname),
        'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
        'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
        'target': int(target)
    }

json.dump(IMAGENET_S_infodicts, open(f'{infd_path}/IMAGENET_S_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/IMAGENET_S_classnames.json', 'w'), indent=4)



#%% ##########################################################    IMAGENET_V2
import data_lib.imagenet_v2
root = './data/IMAGENET_V2'

caption_dict = pkl.load(open('./data/dataset_captions/IMAGENET_V2_captions.pkl', 'rb'))

data = []
targets = []
classes = sorted(os.listdir(os.path.join(root, 'test')), key=lambda x: x.lower())
for i, folder in enumerate(classes):
    for file in os.listdir(os.path.join(root, 'test', folder)):
        data.append(os.path.join(root, 'test', folder, file))
        targets.append(i)

ordered_class_names = sorted(data_lib.imagenet.BASE_CLASSES, key=lambda x: x.lower())

IMAGENET_V2_infodicts = {}
for path, target in zip(data, targets):
    classname = ordered_class_names[target]
    classinfo = '/'.join(path.split('/')[-2:])
    ref_path = f'IMAGENET_V2/IMAGENET_V2_test_224/{classinfo}'
    IMAGENET_V2_infodicts[path.replace(f'{root}/','')] = {
        'classname': classname,
        'default_caption': None,
        'primer_caption': data_lib.imagenet_v2.PRIMER.format(classname),
        'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
        'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
        'target': int(target)
    }

json.dump(IMAGENET_V2_infodicts, open(f'{infd_path}/IMAGENET_V2_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/IMAGENET_V2_classnames.json', 'w'), indent=4)


#%% ##########################################################    iNATURALIST2021
import data_lib.inaturalist2021
root = './data/iNATURALIST2021'
splits = ['train_mini', 'val']

data_dict = {}
target_dict = {}

for split in splits:
    data = []
    targets = []
    datapath = os.path.join(root, split)
    # We only use 25% of the classes.
    classes_to_use = sorted(os.listdir(datapath))[::4]        
    if split == 'train_mini':
        ordered_class_names = classes_to_use
        
    for i, classname in enumerate(classes_to_use):
        classpath = os.path.join(datapath, classname)
        for file in os.listdir(classpath):
            data.append(os.path.join(classpath, file))
            targets.append(i)
    data_dict[split] = data
    target_dict[split] = targets
    
ordered_class_names = [' '.join(x.split('_')[1:]) for x in classes_to_use]

caption_dict = pkl.load(open('./data/dataset_captions/iNATURALIST2021_captions.pkl', 'rb'))

iNATURALIST2021_infodicts = {}

for split in splits:
    iNATURALIST2021_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'iNATURALIST2021/iNATURALIST2021_{split}_224/{classinfo}'
        iNATURALIST2021_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.inaturalist2021.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(iNATURALIST2021_infodicts['train_mini'], open(f'{infd_path}/iNATURALIST2021_train_mini.json', 'w'), indent=4)
json.dump(iNATURALIST2021_infodicts['val'], open(f'{infd_path}/iNATURALIST2021_val.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/iNATURALIST2021_classnames.json', 'w'), indent=4)



#%% ##########################################################    ISICMELANOMA
import pandas as pd
import data_lib.isicmelanoma
root = './data/ISICMELANOMA'
splits = ['train', 'test']

context = pd.read_csv(f'{root}/labels.csv')

diagnosis = np.array(context['diagnosis'])

np.random.seed(0)
amp_idcs = list(np.where(diagnosis == 'atypical melanocytic proliferation')[0])
calm_idcs = list(np.where(diagnosis == 'cafe-au-lait macule')[0])
unknown_idcs = list(sorted(np.random.choice(np.where(diagnosis == 'unknown')[0], 26124, replace=False)))
nevus_idcs = list(sorted(np.random.choice(np.where(diagnosis == 'nevus')[0], 4193, replace=False)))

shared_removals = unknown_idcs + nevus_idcs + amp_idcs + calm_idcs
new_subset = [x for x in tqdm.tqdm(range(len(diagnosis))) if x not in shared_removals]
subset = new_subset

diagnosis = diagnosis[subset]
image_path = np.array([f'{root}/train/{name}.jpg' for name in context['image_name']])[subset]
patient_sex = np.array(list(context['sex']))[subset]
approx_age = np.array(list(context['age_approx']))[subset]
anatomy = np.array(list(context['anatom_site_general_challenge']))[subset]
ben_mal = np.array(list(context['benign_malignant']))[subset]

ordered_class_names = sorted(np.unique(diagnosis))
class_name_to_target = {class_name: i for i, class_name in enumerate(ordered_class_names)}
targets = np.array([class_name_to_target[class_name] for class_name in diagnosis])

ordered_class_names[-1] = 'unknown symptoms'
diagnosis = np.array([x if x != 'unknown' else 'unknown symptoms' for x in diagnosis])

base_cap_variants = [
    "f'A close-up photo of human skin (from the {location}) with a {mtype} mole exhibiting {diag} (approx. {year} year old {sex}).'",
    "f'A detailed photo showcasing human skin from the {location}, featuring a {mtype} mole displaying {diag}. Subject is a {year} year old {sex}.'",
    "f'A close view of the human skin from a {year} year-old {sex} from {location}, showing a mole that is categorized as {mtype}. The condition is {diag}.'",
    "f'Detailed image of skin from the {location} of a {sex} ({year} years old) with a {mtype} mole, diagnosed as {diag}.'",
    "f'Featuring a {diag} in a {mtype} mole, this close-up shot focuses on the skin of a {year} year-old {sex} from the {location}.'",
    "f'Captured from the {location}, this photo shows a {sex} with a {mtype} mole, around {year} years old, under the diagnosis of {diag}.'",
    "f'Approximately {year} years old, the {sex} depicted here has a {mtype} mole from the {location}, which exhibits {diag}.'",
    "f'A {year} year-old {sex} from the {location} displays a {mtype} mole on their skin, diagnosed with {diag}, in this close-up photograph.'",
    "f'A zoomed-in photograph of the skin on the {location}, where a {mtype} mole can be seen. The diagnosis is {diag}, on a {sex} (~{year} years old).'",
    "f'Detailing human skin from {location}, this image focuses on a {mtype} mole with a {diag}. The photograph depicts a {sex} around {year} years old.'"
]
def_cap_variants = [
    "f'A close-up photo of human skin with a mole exhibiting {diag}.'",
    "f'A detailed photo showcasing human skin featuring a mole displaying {diag}.'",
    "f'A close view of human skin showing a mole categorized as {diag}.'",
    "f'Detailed image of skin from a mole diagnosed as {diag}.'",
    "f'Close-up shot of a {diag} mole.'",
    "f'Picture of a mole with {diag}.'",
    "f'A mole exhibiting {diag}.'",
    "f'A photo of a {diag} mole on human skin.'",
    "f'A zoomed-in photograph of the skin, where a mole can be seen. The diagnosis is {diag}.'",
    "f'Detailing human skin with a {diag} mole.'"
]
joint_cap_vars = base_cap_variants + def_cap_variants
captions = []

np.random.seed(0)
for location, mtype, year, diag, sex in zip(anatomy, ben_mal, approx_age, diagnosis, patient_sex):
    if diag == 'unknown':
        diag = 'unknown symptoms'
    if not np.isnan(year) and sex != 'nan' and location != 'nan':
        year = int(year)
        cap_var = np.random.choice(joint_cap_vars)
    else:
        cap_var = np.random.choice(def_cap_variants)
        
    captions.append(eval(cap_var))

captions = np.array(captions)

test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))

image_path_dict = {'train': image_path[train_idcs], 'test': image_path[test_idcs]}
targets_dict = {'train': targets[train_idcs], 'test': targets[test_idcs]}
captions_dict = {'train': captions[train_idcs], 'test': captions[test_idcs]}


ordered_class_names = [
    'lentigo NOS', 'lichenoid keratosis', 'melanoma', 'nevus', 'seborrheic keratosis', 'solar lentigo', 'unknown symptoms'
]

ISICMELANOMA_infodicts = {}

for split in splits:
    ISICMELANOMA_infodicts[split] = {}
    data_list = image_path_dict[split]
    target_list = targets_dict[split]
    cap_list = captions_dict[split]
    for path, target, caption in zip(data_list, target_list, cap_list):
        classname = ordered_class_names[target]
        ISICMELANOMA_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': caption,
            'primer_caption': data_lib.isicmelanoma.PRIMER.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(ISICMELANOMA_infodicts['train'], open(f'{infd_path}/ISICMELANOMA_train.json', 'w'), indent=4)
json.dump(ISICMELANOMA_infodicts['test'], open(f'{infd_path}/ISICMELANOMA_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/ISICMELANOMA_classnames.json', 'w'), indent=4)



#%% ##########################################################    MEDMNIST_DERMA
import data_lib.medmnist_derma
import medmnist
root = './data/MedMNISTderma'
splits = ['train', 'test']
datasets = {
    'train': medmnist.DermaMNIST(root=root, split='train', download=True),
    'test': medmnist.DermaMNIST(root=root, split='test', download=True)
}

ordered_class_names = [datasets['train'].info['label'][str(x)] for x in range(7)]

MedMNISTderma_caption_dict = pkl.load(open('./data/dataset_captions/MedMNISTderma_captions.pkl', 'rb'))

MedMNISTderma_infodicts = {}

for split in splits:
    data = list(range(len(datasets[split].imgs)))
    targets = np.array(datasets[split].labels.reshape(-1))
    classnames = []
    for target in targets:
        classnames.append(ordered_class_names[target])

    MedMNISTderma_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        key_path = f'medmnistderma-images-{split}/{target}-{path}.png'        
        MedMNISTderma_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.medmnist_derma.PRIMER.format(classname),
            'synthetic_caption': MedMNISTderma_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': MedMNISTderma_caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }
    MedMNISTderma_infodicts[split] = MedMNISTderma_infodict

json.dump(MedMNISTderma_infodicts['train'], open(f'{infd_path}/MedMNISTderma_train.json', 'w'), indent=4)
json.dump(MedMNISTderma_infodicts['test'], open(f'{infd_path}/MedMNISTderma_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MedMNISTderma_classnames.json', 'w'), indent=4)



#%% ##########################################################    MEDMNIST_ORGANC
import data_lib.medmnist_organc
import medmnist
root = './data/MedMNISTorganc'
splits = ['train', 'test']
datasets = {
    'train': medmnist.OrganCMNIST(root=root, split='train', download=True),
    'test': medmnist.OrganCMNIST(root=root, split='test', download=True)
}

ordered_class_names = [datasets['train'].info['label'][str(x)] for x in range(11)]

MedMNISTorganc_caption_dict = pkl.load(open('./data/dataset_captions/MedMNISTorganc_captions.pkl', 'rb'))

MedMNISTorganc_infodicts = {}

for split in splits:
    data = list(range(len(datasets[split].imgs)))
    targets = np.array(datasets[split].labels.reshape(-1))
    classnames = []
    for target in targets:
        classnames.append(ordered_class_names[target])

    MedMNISTorganc_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        key_path = f'medmnistorganc-images-{split}/{target}-{path}.png'        
        MedMNISTorganc_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.medmnist_organc.PRIMER.format(classname),
            'synthetic_caption': MedMNISTorganc_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': MedMNISTorganc_caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }
    MedMNISTorganc_infodicts[split] = MedMNISTorganc_infodict

ordered_class_names = [x.replace('-', ' ') for x in ordered_class_names]
json.dump(MedMNISTorganc_infodicts['train'], open(f'{infd_path}/MedMNISTorganc_train.json', 'w'), indent=4)
json.dump(MedMNISTorganc_infodicts['test'], open(f'{infd_path}/MedMNISTorganc_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MedMNISTorganc_classnames.json', 'w'), indent=4)



#%% ##########################################################    MEDMNIST_ORGANS
import data_lib.medmnist_organs
import medmnist
root = './data/MedMNISTorgans'
splits = ['train', 'test']
datasets = {
    'train': medmnist.OrganSMNIST(root=root, split='train', download=True),
    'test': medmnist.OrganSMNIST(root=root, split='test', download=True)
}

ordered_class_names = [datasets['train'].info['label'][str(x)] for x in range(11)]

MedMNISTorgans_caption_dict = pkl.load(open('./data/dataset_captions/MedMNISTorgans_captions.pkl', 'rb'))

MedMNISTorgans_infodicts = {}

for split in splits:
    data = list(range(len(datasets[split].imgs)))
    targets = np.array(datasets[split].labels.reshape(-1))
    classnames = []
    for target in targets:
        classnames.append(ordered_class_names[target])

    MedMNISTorgans_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        key_path = f'medmnistorgans-images-{split}/{target}-{path}.png'        
        MedMNISTorgans_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.medmnist_organc.PRIMER.format(classname),
            'synthetic_caption': MedMNISTorgans_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': MedMNISTorgans_caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }
    MedMNISTorgans_infodicts[split] = MedMNISTorgans_infodict

ordered_class_names = [x.replace('-', ' ') for x in ordered_class_names]
json.dump(MedMNISTorgans_infodicts['train'], open(f'{infd_path}/MedMNISTorgans_train.json', 'w'), indent=4)
json.dump(MedMNISTorgans_infodicts['test'], open(f'{infd_path}/MedMNISTorgans_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MedMNISTorgans_classnames.json', 'w'), indent=4)

#%% ##########################################################    MITSTATES
import data_lib.mitstates
root = './data/MITSTATES'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

for split in splits:
    data = []
    targets = []
    classes = sorted(os.listdir(f'{root}/data/images'))
    classes = [x for x in classes if 'adj ' not in x and 'DS_Store' not in x]
    for i, folder in enumerate(classes):
        for file in os.listdir(os.path.join(root, 'data', 'images', folder)):
            data.append(os.path.join(root, 'data', 'images', folder, file))
            targets.append(i)
                                
    test_idcs = range(0, len(targets), 5)
    train_idcs = list(set(range(len(targets))) - set(test_idcs))
    subset = train_idcs if split == 'train' else test_idcs
    data_dict[split] = [data[i] for i in subset]
    target_dict[split] = [targets[i] for i in subset]
    
caption_dict = pkl.load(open('./data/dataset_captions/MITSTATES_captions.pkl', 'rb'))
ordered_class_names = classes

MITSTATES_infodicts = {}

for split in splits:
    MITSTATES_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        path = path.replace(f'{root}/','')
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'MITSTATES/{path}'
        MITSTATES_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.mitstates.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(MITSTATES_infodicts['train'], open(f'{infd_path}/MITSTATES_train.json', 'w'), indent=4)
json.dump(MITSTATES_infodicts['test'], open(f'{infd_path}/MITSTATES_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MITSTATES_classnames.json', 'w'), indent=4)




#%% ##########################################################    MNIST
import data_lib.mnist
root = './data/MNIST'
splits = ['train', 'test']
datasets = {
    'train': torchvision.datasets.MNIST(root, True, transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.MNIST(root, False, transform=None, target_transform=None, download=True)
}
ordered_class_names = [x.split(' -')[0] for x in list(datasets['train'].classes)]
mnist_caption_dict = pkl.load(open('./data/dataset_captions/MNIST_captions.pkl', 'rb'))

mnist_infodicts = {}

for split in splits:
    data = list(range(len(datasets[split].data)))
    targets = np.array(datasets[split].targets)
    conversion = {val:key for key, val in datasets[split].class_to_idx.items()}
    classnames = []
    for target in targets:
        classnames.append(conversion[target])

    mnist_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        key_path = f'mnist-images-{split}/{target}-{path}.png'        
        mnist_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.mnist.PRIMER.format(classname),
            'synthetic_caption': mnist_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': mnist_caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }
    mnist_infodicts[split] = mnist_infodict

json.dump(mnist_infodicts['train'], open(f'{infd_path}/MNIST_train.json', 'w'), indent=4)
json.dump(mnist_infodicts['test'], open(f'{infd_path}/MNIST_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MNIST_classnames.json', 'w'), indent=4)





#%% ##########################################################    MONKEYS10
import pandas as pd
import data_lib.monkeys10
root = './data/MONKEYS10'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

for split in splits:
    data = []
    targets = []

    label_df = pd.read_csv(os.path.join(root, 'monkeys10', 'monkey_labels.txt'), skipinitialspace=True).to_numpy()
    i2c_mapping = {int(l[0].strip()[-1]): l[2].strip().replace('_', ' ') for l in label_df}
    c2i_mapping = {v:k for k,v in i2c_mapping.items()}

    data = []
    targets = []

    target_folder = os.path.join(root, 'monkeys10', 'training', 'training') if split == 'train' else os.path.join(root, 'monkeys10', 'validation', 'validation')

    for index, folder in enumerate(sorted(os.listdir(target_folder))):
        for file in os.listdir(os.path.join(target_folder, folder)):
            # remove all corrupted files / non image files
            if '.jpg' in file or '.png' in file:
                data.append(os.path.join(target_folder, folder, file))
                targets.append(index)

    data_dict[split] = data
    target_dict[split] = targets
   
caption_dict = pkl.load(open('./data/dataset_captions/MONKEYS10_captions.pkl', 'rb'))
ordered_class_names = [i2c_mapping[i] for i in range(10)]

MONKEYS10_infodicts = {}

for split in splits:
    MONKEYS10_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        path = path.replace(f'{root}/','')
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'MONKEYS10/MONKEYS10_{split}_224/{classinfo}'
        MONKEYS10_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.monkeys10.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(MONKEYS10_infodicts['train'], open(f'{infd_path}/MONKEYS10_train.json', 'w'), indent=4)
json.dump(MONKEYS10_infodicts['test'], open(f'{infd_path}/MONKEYS10_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MONKEYS10_classnames.json', 'w'), indent=4)




#%% ##########################################################    MSCOCO
import data_lib.mscoco
import pandas as pd

root = './data/MSCOCO'
metadata = pd.read_csv(f'{root}/annotations.csv')                
data = [f'{root}/images_mscoco_2014_5k_test/{file}' for file, split in zip(metadata.filename, metadata.split) if split == 'test']
caption_data = [eval(x) for x, split in zip(metadata.raw, metadata.split) if split == 'test']
# For retrieval datasets, targets are simply the associated caption idcs.
targets = torch.arange(len(data)).to(torch.long)

mscoco_infodict = {}
for path, caption, target in zip(data, caption_data, targets):        
    mscoco_infodict[path.replace('./data/MSCOCO/','')] = {
        'classname': None,
        'default_caption': caption,
        'primer_caption': None,
        'synthetic_caption': None,
        'synthetic_merged_caption': None,
        'target': int(target)
    }

json.dump(mscoco_infodict, open(f'{infd_path}/MSCOCO_test.json', 'w'), indent=4)



#%% ##########################################################    MTSD
import data_lib.mtsd
root = './data/MTSD'
splits = ['train', 'val']

ordered_class_names = [
    'accident-area', 'accidental-area-unsure', 'added-lane-right', 'airport', 'bicycles-crossing', 'bicycles-only', 'bike-route', 'both-directions', 'bus-stop', 
    'bus-stop-ahead', 'buses', 'buses-only', 'camp', 'chevron-left', 'chevron-right', 'chevron-right-unsure', 'children', 'crossroads', 
    'crossroads-with-priority-to-the-right', 'curve-left', 'curve-right', 'dead-end', 'detour-left', 'dip', 'disabled-persons', 'distance', 'divided-highway-ends', 
    'do-not-block-intersection', 'do-not-stop-on-tracks', 'domestic-animals', 'double-curve-first-left', 'double-curve-first-right', 'double-reverse-curve-right', 
    'double-turn-first-right', 'dual-lanes-go-straight-on-left', 'dual-lanes-go-straight-on-right', 'dual-lanes-right-turn-or-go-straight', 
    'dual-lanes-turn-left-no-u-turn', 'dual-lanes-turn-left-or-straight', 'dual-path-bicycles-and-pedestrians', 'dual-path-pedestrians-and-bicycles', 'emergency-facility', 
    'emergency-vehicles', 'end-of-bicycles-only', 'end-of-built-up-area', 'end-of-buses-only', 'end-of-limited-access-road', 'end-of-living-street', 
    'end-of-maximum-speed-limit-30', 'end-of-maximum-speed-limit-70', 'end-of-motorway', 'end-of-no-parking', 'end-of-pedestrians-only', 'end-of-priority-road', 
    'end-of-prohibition', 'end-of-speed-limit-zone', 'equestrians-crossing', 'except-bicycles', 'extent-of-prohibition-area-both-direction', 
    'falling-rocks-or-debris-right', 'flaggers-in-road', 'food', 'gas-station', 'give-way-to-oncoming-traffic', 'go-left', 'go-right', 'go-straight', 
    'go-straight-or-turn-left', 'go-straight-or-turn-right', 'hairpin-curve-left', 'hairpin-curve-right', 'height-limit', 'height-restriction', 'highway-exit', 
    'highway-interstate-route', 'horizontal-alignment-left', 'horizontal-alignment-right', 'hospital', 'interstate-route', 'junction-with-a-side-road-acute-left', 
    'junction-with-a-side-road-acute-right', 'junction-with-a-side-road-perpendicular-left', 'junction-with-a-side-road-perpendicular-right', 'kangaloo-crossing', 
    'keep-left', 'keep-right', 'lane-control', 'left-turn-yield-on-green', 'limited-access-road', 'living-street', 'lodging', 'loop-270-degree', 'maximum-speed-limit-10', 
    'maximum-speed-limit-100', 'maximum-speed-limit-110', 'maximum-speed-limit-120', 'maximum-speed-limit-15', 'maximum-speed-limit-20', 'maximum-speed-limit-25', 
    'maximum-speed-limit-30', 'maximum-speed-limit-35', 'maximum-speed-limit-40', 'maximum-speed-limit-45', 'maximum-speed-limit-5', 'maximum-speed-limit-50', 
    'maximum-speed-limit-55', 'maximum-speed-limit-60', 'maximum-speed-limit-65', 'maximum-speed-limit-70', 'maximum-speed-limit-75', 'maximum-speed-limit-80', 
    'maximum-speed-limit-90', 'maximum-speed-limit-led-100', 'maximum-speed-limit-led-60', 'maximum-speed-limit-led-80', 'minimum-safe-distance', 'minimum-speed-40', 
    'mopeds-and-bicycles-only', 'motorway', 'narrow-bridge', 'no-bicycles', 'no-buses', 'no-entry', 'no-hawkers', 'no-heavy-goods-vehicles', 
    'no-heavy-goods-vehicles-or-buses', 'no-left-turn', 'no-mopeds-or-bicycles', 'no-motor-vehicle-trailers', 'no-motor-vehicles', 'no-motor-vehicles-except-motorcycles', 
    'no-motorcycles', 'no-overtaking', 'no-overtaking-by-heavy-goods-vehicles', 'no-parking', 'no-parking-or-no-stopping', 'no-pedestrians', 'no-pedestrians-or-bicycles', 
    'no-right-turn', 'no-stopping', 'no-straight-through', 'no-turn-on-red', 'no-turns', 'no-u-turn', 'no-vehicles-carrying-dangerous-goods', 'obstacle-delineator', 
    'offset-roads', 'one-direction-left', 'one-direction-right', 'one-way-left', 'one-way-right', 'one-way-straight', 'other-danger', 'parking', 'parking-restrictions', 
    'pass-left-or-right', 'pass-on-either-side', 'pass-right', 'passing-lane-ahead', 'pedestrian-stumble-train', 'pedestrians-crossing', 'pedestrians-only', 'playground', 
    'priority-over-oncoming-vehicles', 'priority-road', 'priority-route-at-intersection', 'radar-enforced', 'railroad-crossing', 'railroad-crossing-with-barriers', 
    'railroad-crossing-without-barriers', 'railroad-intersection', 'reversible-lanes', 'road-bump', 'road-closed', 'road-closed-to-vehicles', 'road-narrows', 
    'road-narrows-left', 'road-narrows-right', 'road-widens', 'road-widens-right', 'roadworks', 'roundabout', 'safety-area', 'school-zone', 
    'shared-path-bicycles-and-pedestrians', 'shared-path-pedestrians-and-bicycles', 'slippery-motorcycles', 'slippery-road-surface', 'steep-ascent', 'stop', 'stop-ahead', 
    'stop-here-on-red-or-flashing-light', 'stop-signals', 't-roads', 'telephone', 'text-four-lines', 'texts', 'tow-away-zone', 'traffic-merges-left', 
    'traffic-merges-right', 'traffic-signals', 'trail-crossing', 'trailer-camping', 'tram-bus-stop', 'trams-crossing', 'triple-lanes-turn-left-center-lane', 'trucks', 
    'trucks-crossing', 'trucks-turn-right', 'turn-left', 'turn-left-ahead', 'turn-right', 'turn-right-ahead', 'turning-vehicles-yield-to-pedestrians', 'two-way-traffic', 
    'u-turn', 'uneven-road', 'uneven-roads-ahead', 'weight-limit', 'width-limit', 'wild-animals', 'winding-road-first-left', 'winding-road-first-right', 'wombat-crossing', 
    'wrong-way', 'y-roads', 'yield'
]

data_dict = {}
target_dict = {}

for split in splits:
    data = []
    targets = []
    classnames = []
    base_path = os.path.join(root, 'MTSD', 'processed', split)        
    count = 0
    for i, folder in enumerate(sorted(os.listdir(base_path))):
        if folder in ordered_class_names:
            classnames.append(folder)            
            for file in sorted(os.listdir(os.path.join(base_path, folder))):
                data.append(os.path.join(base_path, folder, file))
                targets.append(count)
            count += 1
    data_dict[split] = data
    target_dict[split] = targets
    
caption_dict = pkl.load(open('./data/dataset_captions/MTSD_captions.pkl', 'rb'))

MTSD_infodicts = {}

for split in splits:
    MTSD_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'MTSD/MTSD/processed/{split}/{classinfo}'
        MTSD_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.mtsd.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(MTSD_infodicts['train'], open(f'{infd_path}/MTSD_train.json', 'w'), indent=4)
json.dump(MTSD_infodicts['val'], open(f'{infd_path}/MTSD_val.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MTSD_classnames.json', 'w'), indent=4)

        
#%% ##########################################################    MVTEC_Adapt
import data_lib.mvtecad_adapt
root = './data/MVTECAD_Adapt'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

ordered_class_names = sorted([x for x in os.listdir(os.path.join(root, 'data')) if x != 'README.txt'])

data = []
targets = []
for i, folder in enumerate(ordered_class_names):
    for file in sorted(os.listdir(os.path.join(root, 'data', folder))):
        data.append(os.path.join(root, 'data', folder, file))
        targets.append(i)
test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))
        
for split in splits:    
    subset = train_idcs if split == 'train' else test_idcs
    data_dict[split] = np.array([data[i] for i in subset])
    target_dict[split] = np.array([targets[i] for i in subset])

caption_dict = pkl.load(open('./data/dataset_captions/MVTECAD_Adapt_captions.pkl', 'rb'))
caption_dict = {'/'.join(key.split('/')[-2:]): item for key, item in caption_dict.items()}

MVTECAD_Adapt_infodicts = {}

for split in splits:
    MVTECAD_Adapt_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = classinfo
        MVTECAD_Adapt_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.mvtecad_adapt.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(MVTECAD_Adapt_infodicts['train'], open(f'{infd_path}/MVTECAD_Adapt_train.json', 'w'), indent=4)
json.dump(MVTECAD_Adapt_infodicts['test'], open(f'{infd_path}/MVTECAD_Adapt_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MVTECAD_Adapt_classnames.json', 'w'), indent=4)



#%% ##########################################################    MVTEC_Defect
import data_lib.mvtecad_eval
root = './data/MVTECAD_Eval'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

ordered_class_names = sorted([x for x in os.listdir(os.path.join(root, 'data')) if x != 'README.txt'])

data = []
targets = []
for i, folder in enumerate(ordered_class_names):
    for file in sorted(os.listdir(os.path.join(root, 'data', folder))):
        data.append(os.path.join(root, 'data', folder, file))
        targets.append(i)
test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))
        
for split in splits:    
    subset = train_idcs if split == 'train' else test_idcs
    data_dict[split] = np.array([data[i] for i in subset])
    target_dict[split] = np.array([targets[i] for i in subset])

caption_dict = pkl.load(open('./data/dataset_captions/MVTECAD_Eval_captions.pkl', 'rb'))
caption_dict = {'/'.join(key.split('/')[-2:]): item for key, item in caption_dict.items()}

import copy
old_ordered_class_names = copy.deepcopy(ordered_class_names)

a = [x.replace('-good','').replace('-', ' with ').replace('_', ' ') for x in ordered_class_names]
for i in range(len(a)):
    a[i] = a[i].replace('broken large', 'large break')
    a[i] = a[i].replace('broken small', 'small break')
    a[i] = a[i].replace('cable swap', 'inner cable swap')
    a[i] = a[i].replace('missing cable', 'missing inner cable')
    a[i] = a[i].replace('with color', 'with colored spots')
    a[i] = a[i].replace('scratch', 'scratched')
    a[i] = a[i].replace('toothbrush with defective', 'defective toothbrush')
    a[i] = a[i].replace('transistor with misplaced', 'missing transistor')
ordered_class_names = a


MVTECAD_Eval_infodicts = {}

backward_compatibility_converter = {key: value for key, value in zip(old_ordered_class_names, ordered_class_names)}

for split in splits:
    MVTECAD_Eval_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = classinfo
        MVTECAD_Eval_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.mvtecad_eval.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }


json.dump(MVTECAD_Eval_infodicts['train'], open(f'{infd_path}/MVTECAD_Eval_train.json', 'w'), indent=4)
json.dump(MVTECAD_Eval_infodicts['test'], open(f'{infd_path}/MVTECAD_Eval_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/MVTECAD_Eval_classnames.json', 'w'), indent=4)


#%% ##########################################################    ObjectNet
import data_lib.objectnet
root = './data/OBJECTNET'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

folder_names = [
    'air_freshener', 'alarm_clock', 'backpack', 'baking_sheet', 'banana', 'band_aid', 'baseball_bat', 'baseball_glove', 
    'basket', 'bathrobe', 'battery', 'bed_sheet', 'beer_bottle', 'beer_can', 'belt', 'bench', 'bicycle', 'bike_pump', 'bills_money', 
    'binder_closed', 'biscuits', 'blanket', 'blender', 'blouse', 'board_game', 'book_closed', 'bookend', 'boots', 'bottle_cap', 
    'bottle_opener', 'bottle_stopper', 'box', 'bracelet', 'bread_knife', 'bread_loaf', 'briefcase', 'brooch', 'broom', 'bucket', 
    'butchers_knife', 'butter', 'button', 'calendar', 'can_opener', 'candle', 'canned_food', 'cd_case', 'cellphone', 'cellphone_case', 
    'cellphone_charger', 'cereal', 'chair', 'cheese', 'chess_piece', 'chocolate', 'chopstick', 'clothes_hamper', 'clothes_hanger', 'coaster', 
    'coffee_beans', 'coffee_french_press', 'coffee_grinder', 'coffee_machine', 'coffee_table', 'coin_money', 'comb', 'combination_lock', 
    'computer_mouse', 'contact_lens_case', 'cooking_oil_bottle', 'cork', 'cutting_board', 'deodorant', 'desk_lamp', 'detergent', 'dish_soap', 
    'document_folder_closed', 'dog_bed', 'doormat', 'drawer_open', 'dress', 'dress_pants', 'dress_shirt', 'dress_shoe_men', 'dress_shoe_women', 
    'drill', 'drinking_cup', 'drinking_straw', 'drying_rack_for_clothes', 'drying_rack_for_dishes', 'dust_pan', 'dvd_player', 'earbuds', 'earring', 
    'egg', 'egg_carton', 'envelope', 'eraser_white_board', 'extension_cable', 'eyeglasses', 'fan', 'figurine_or_statue', 'first_aid_kit', 
    'flashlight', 'floss_container', 'flour_container', 'fork', 'frying_pan', 'full_sized_towel', 'glue_container', 'hair_brush', 'hair_dryer', 
    'hairclip', 'hairtie', 'hammer', 'hand_mirror', 'hand_towel_or_rag', 'handbag', 'hat', 'headphones_over_ear', 'helmet', 'honey_container', 
    'ice', 'ice_cube_tray', 'iron_for_clothes', 'ironing_board', 'jam', 'jar', 'jeans', 'kettle', 'key_chain', 'keyboard', 'ladle', 'lampshade', 
    'laptop_charger', 'laptop_open', 'leaf', 'leggings', 'lemon', 'letter_opener', 'lettuce', 'light_bulb', 'lighter', 'lipstick', 'loofah', 
    'magazine', 'makeup', 'makeup_brush', 'marker', 'match', 'measuring_cup', 'microwave', 'milk', 'mixing_salad_bowl', 'monitor', 'mouse_pad', 
    'mouthwash', 'mug', 'multitool', 'nail_clippers', 'nail_fastener', 'nail_file', 'nail_polish', 'napkin', 'necklace', 'newspaper', 'night_light', 
    'nightstand', 'notebook', 'notepad', 'nut_for_screw', 'orange', 'oven_mitts', 'padlock', 'paint_can', 'paintbrush', 'paper', 'paper_bag', 'paper_plates', 
    'paper_towel', 'paperclip', 'peeler', 'pen', 'pencil', 'pepper_shaker', 'pet_food_container', 'phone_landline', 'photograph_printed', 'pill_bottle', 
    'pill_organizer', 'pillow', 'pitcher', 'placemat', 'plastic_bag', 'plastic_cup', 'plastic_wrap', 'plate', 'playing_cards', 'pliers', 'plunger', 'pop_can', 
    'portable_heater', 'poster', 'power_bar', 'power_cable', 'printer', 'raincoat', 'rake', 'razor', 'receipt', 'remote_control', 'removable_blade', 'ribbon', 
    'ring', 'rock', 'rolling_pin', 'ruler', 'running_shoe', 'safety_pin', 'salt_shaker', 'sandal', 'scarf', 'scissors', 'screw', 'scrub_brush', 'sewing_kit', 
    'shampoo_bottle', 'shoelace', 'shorts', 'shovel', 'skateboard', 'skirt', 'sleeping_bag', 'slipper', 'soap_bar', 'soap_dispenser', 'sock', 'soup_bowl', 
    'spatula', 'speaker', 'sponge', 'spoon', 'spray_bottle', 'squeegee', 'squeeze_bottle', 'standing_lamp', 'stapler', 'step_stool', 'still_camera', 
    'stopper_sink_tub', 'strainer', 'stuffed_animal', 'sugar_container', 'suit_jacket', 'suitcase', 'sunglasses', 'sweater', 'swimming_trunks', 't-shirt', 
    'table_knife', 'tablecloth', 'tablet_ipad', 'tanktop', 'tape', 'tape_measure', 'tarp', 'teabag', 'teapot', 'tennis_racket', 'thermometer', 'thermos', 
    'throw_pillow', 'tie', 'tissue', 'toaster', 'toilet_paper_roll', 'tomato', 'tongs', 'toothbrush', 'toothpaste', 'tote_bag', 'toy', 'trash_bag', 
    'trash_bin', 'travel_case', 'tray', 'trophy', 'tv', 'tweezers', 'umbrella', 'usb_cable', 'usb_flash_drive', 'vacuum_cleaner', 'vase', 'video_camera', 
    'walker', 'walking_cane', 'wallet', 'watch', 'water_bottle', 'water_filter', 'webcam', 'weight_exercise', 'weight_scale', 'wheel', 'whisk', 'whistle', 
    'wine_bottle', 'wine_glass', 'winter_glove', 'wok', 'wrench', 'ziploc_bag'
]

ordered_class_names = [x.replace('_', ' ') for x in folder_names]

all_classes_root = os.path.join(root, 'datasets', 'objectnet-1.0-beta')
classnames = os.listdir(all_classes_root)
caption_dict = pkl.load(open('./data/dataset_captions/OBJECTNET_captions.pkl', 'rb'))

data_dict = {}
target_dict = {}

data = []
targets = []
cls_coll = []
for folder in classnames:
    if '.DS' not in folder:
        files = [os.path.join(all_classes_root, folder, x) for x in os.listdir(os.path.join(all_classes_root, folder)) if '.DS' not in x]
        targs = [folder_names.index(folder)] * len(files)
        data += files
        targets += targs
        classname = ordered_class_names[folder_names.index(folder)]
        cls_coll += [classname for _ in range(len(files))]

data = np.array(data)
targets = np.array(targets)
cls_coll = np.array(cls_coll)

test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))

data_dict = {'train': data[train_idcs], 'test': data[test_idcs]}
targets_dict = {'train': targets[train_idcs], 'test': targets[test_idcs]}
cls_coll_dict = {'train': cls_coll[train_idcs], 'test': cls_coll[test_idcs]}

tar_to_cls = {}
for target, classname in zip(targets, cls_coll):
    tar_to_cls[target] = classname
    
OBJECTNET_infodicts = {}

for split in splits:
    OBJECTNET_infodicts[split] = {}
    data_list = data_dict[split]
    targets = targets_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'OBJECTNET/OBJECTNET_test_224/{classinfo}'
        OBJECTNET_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.objectnet.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(OBJECTNET_infodicts['train'], open(f'{infd_path}/OBJECTNET_train.json', 'w'), indent=4)
json.dump(OBJECTNET_infodicts['test'], open(f'{infd_path}/OBJECTNET_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/OBJECTNET_classnames.json', 'w'), indent=4)



#%% ##########################################################    OBSC_ANIMALS
import data_lib.obsc_animals
root = './data/OBSC_ANIMALS'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

split_dict = json.load(open(os.path.join(root, 'split.json'), 'r'))

for split in splits:
    if split == 'train':
        split_sub_dict = split_dict['train'] + split_dict['val']
    else:
        split_sub_dict = split_dict['test']

    data_dict[split] = np.array([os.path.join(root, x[0]) for x in split_sub_dict])
    target_dict[split] = np.array([x[1] for x in split_sub_dict])
    
    if split == 'train':
        classnames = {}
        for x in split_sub_dict:
            classnames[x[1]] = x[2]

caption_dict = pkl.load(open('./data/dataset_captions/OBSC_ANIMALS_captions.pkl', 'rb'))

ordered_class_names = [classnames[i] for i in range(len(classnames))]

OBSC_ANIMALS_infodicts = {}

for split in splits:
    OBSC_ANIMALS_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = classnames[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'OBSC_ANIMALS/{classinfo}'
        OBSC_ANIMALS_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.obsc_animals.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }


json.dump(OBSC_ANIMALS_infodicts['train'], open(f'{infd_path}/OBSC_ANIMALS_train.json', 'w'), indent=4)
json.dump(OBSC_ANIMALS_infodicts['test'], open(f'{infd_path}/OBSC_ANIMALS_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/OBSC_ANIMALS_classnames.json', 'w'), indent=4)


#%% ##########################################################    OBSC_THINGS
import data_lib.obsc_things
root = './data/OBSC_THINGS'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

split_dict = json.load(open(os.path.join(root, 'split.json'), 'r'))

for split in splits:
    if split == 'train':
        split_sub_dict = split_dict['train'] + split_dict['val']
    else:
        split_sub_dict = split_dict['test']

    data_dict[split] = np.array([os.path.join(root, x[0]) for x in split_sub_dict])
    target_dict[split] = np.array([x[1] for x in split_sub_dict])
    
    if split == 'train':
        classnames = {}
        for x in split_sub_dict:
            classnames[x[1]] = x[2]

caption_dict = pkl.load(open('./data/dataset_captions/OBSC_THINGS_captions.pkl', 'rb'))

ordered_class_names = [classnames[i] for i in range(len(classnames))]

OBSC_THINGS_infodicts = {}

for split in splits:
    OBSC_THINGS_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = classnames[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'OBSC_THINGS/{classinfo}'
        OBSC_THINGS_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.obsc_things.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }


json.dump(OBSC_THINGS_infodicts['train'], open(f'{infd_path}/OBSC_THINGS_train.json', 'w'), indent=4)
json.dump(OBSC_THINGS_infodicts['test'], open(f'{infd_path}/OBSC_THINGS_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/OBSC_THINGS_classnames.json', 'w'), indent=4)

#%% ##########################################################    OXFORD_PETS
import data_lib.oxford_pets
root = './data/OXFORDPETS'
splits = ['train', 'test']

OXFORDPETS_datasets = {
    'train': torchvision.datasets.OxfordIIITPet(root, split='trainval', transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.OxfordIIITPet(root, split='test', transform=None, target_transform=None, download=True)        
}
caption_dict = pkl.load(open('./data/dataset_captions/OXFORDPETS_captions.pkl', 'rb'))
data_dict = {split: [str(x).replace('data/OXFORDPETS/','') for x in OXFORDPETS_datasets[split]._images] for split in splits}
target_dict = {split: OXFORDPETS_datasets[split]._labels for split in splits}
ordered_class_names = [x.replace('_', ' ') for x in OXFORDPETS_datasets['train'].classes]

OXFORDPETS_infodicts = {}
for split in splits:
    OXFORDPETS_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'OXFORDPETS/{path}'
        OXFORDPETS_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.oxford_pets.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(OXFORDPETS_infodicts['train'], open(f'{infd_path}/OXFORDPETS_train.json', 'w'), indent=4)
json.dump(OXFORDPETS_infodicts['test'], open(f'{infd_path}/OXFORDPETS_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/OXFORDPETS_classnames.json', 'w'), indent=4)


#%% ##########################################################    PLACES365
import pandas as pd
import data_lib.places365
root = './data/PLACES365'
splits = ['train', 'test']
caption_dict = pkl.load(open('./data/dataset_captions/PLACES365_captions.pkl', 'rb'))

ordered_class_names = [
    'airfield', 'airplane cabin', 'airport terminal', 'alcove', 'alley', 'amphitheater', 'amusement arcade', 'amusement park', 
    'apartment building outdoor', 'aquarium', 'aqueduct', 'arcade', 'arch', 'archaelogical excavation', 'archive', 'arena hockey', 
    'arena performance', 'arena rodeo', 'army base', 'art gallery', 'art school', 'art studio', 'artists loft', 'assembly line', 
    'athletic field outdoor', 'atrium public', 'attic', 'auditorium', 'auto factory', 'auto showroom', 'badlands', 'bakery shop', 
    'balcony exterior', 'balcony interior', 'ball pit', 'ballroom', 'bamboo forest', 'bank vault', 'banquet hall', 'bar', 'barn', 
    'barndoor', 'baseball field', 'basement', 'basketball court indoor', 'bathroom', 'bazaar indoor', 'bazaar outdoor', 'beach', 
    'beach house', 'beauty salon', 'bedchamber', 'bedroom', 'beer garden', 'beer hall', 'berth', 'biology laboratory', 'boardwalk', 
    'boat deck', 'boathouse', 'bookstore', 'booth indoor', 'botanical garden', 'bow window indoor', 'bowling alley', 'boxing ring', 
    'bridge', 'building facade', 'bullring', 'burial chamber', 'bus interior', 'bus station indoor', 'butchers shop', 'butte', 'cabin outdoor', 
    'cafeteria', 'campsite', 'campus', 'canal natural', 'canal urban', 'candy store', 'canyon', 'car interior', 'carrousel', 'castle', 'catacomb', 
    'cemetery', 'chalet', 'chemistry lab', 'childs room', 'church indoor', 'church outdoor', 'classroom', 'clean room', 'cliff', 'closet', 
    'clothing store', 'coast', 'cockpit', 'coffee shop', 'computer room', 'conference center', 'conference room', 'construction site', 
    'corn field', 'corral', 'corridor', 'cottage', 'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk', 'dam', 'delicatessen', 
    'department store', 'desert sand', 'desert vegetation', 'desert road', 'diner outdoor', 'dining hall', 'dining room', 'discotheque', 
    'doorway outdoor', 'dorm room', 'downtown', 'dressing room', 'driveway', 'drugstore', 'elevator door', 'elevator lobby', 'elevator shaft', 
    'embassy', 'engine room', 'entrance hall', 'escalator indoor', 'excavation', 'fabric store', 'farm', 'fastfood restaurant', 'field cultivated', 
    'field wild', 'field road', 'fire escape', 'fire station', 'fishpond', 'flea market indoor', 'florist shop indoor', 'food court', 
    'football field', 'forest broadleaf', 'forest path', 'forest road', 'formal garden', 'fountain', 'galley', 'garage indoor', 'garage outdoor', 
    'gas station', 'gazebo exterior', 'general store indoor', 'general store outdoor', 'gift shop', 'glacier', 'golf course', 'greenhouse indoor', 
    'greenhouse outdoor', 'grotto', 'gymnasium indoor', 'hangar indoor', 'hangar outdoor', 'harbor', 'hardware store', 'hayfield', 'heliport', 
    'highway', 'home office', 'home theater', 'hospital', 'hospital room', 'hot spring', 'hotel outdoor', 'hotel room', 'house', 
    'hunting lodge outdoor', 'ice cream parlor', 'ice floe', 'ice shelf', 'ice skating rink indoor', 'ice skating rink outdoor', 'iceberg', 
    'igloo', 'industrial area', 'inn outdoor', 'islet', 'jacuzzi indoor', 'jail cell', 'japanese garden', 'jewelry shop', 'junkyard', 'kasbah', 
    'kennel outdoor', 'kindergarden classroom', 'kitchen', 'lagoon', 'lake natural', 'landfill', 'landing deck', 'laundromat', 'lawn', 
    'lecture room', 'legislative chamber', 'library indoor', 'library outdoor', 'lighthouse', 'living room', 'loading dock', 'lobby', 
    'lock chamber', 'locker room', 'mansion', 'manufactured home', 'market indoor', 'market outdoor', 'marsh', 'martial arts gym', 'mausoleum', 
    'medina', 'mezzanine', 'moat water', 'mosque outdoor', 'motel', 'mountain', 'mountain path', 'mountain snowy', 'movie theater indoor', 
    'museum indoor', 'museum outdoor', 'music studio', 'natural history museum', 'nursery', 'nursing home', 'oast house', 'ocean', 'office', 
    'office building', 'office cubicles', 'oilrig', 'operating room', 'orchard', 'orchestra pit', 'pagoda', 'palace', 'pantry', 'park', 
    'parking garage indoor', 'parking garage outdoor', 'parking lot', 'pasture', 'patio', 'pavilion', 'pet shop', 'pharmacy', 'phone booth', 
    'physics laboratory', 'picnic area', 'pier', 'pizzeria', 'playground', 'playroom', 'plaza', 'pond', 'porch', 'promenade', 'pub indoor', 
    'racecourse', 'raceway', 'raft', 'railroad track', 'rainforest', 'reception', 'recreation room', 'repair shop', 'residential neighborhood', 
    'restaurant', 'restaurant kitchen', 'restaurant patio', 'rice paddy', 'river', 'rock arch', 'roof garden', 'rope bridge', 'ruin', 'runway', 
    'sandbox', 'sauna', 'schoolhouse', 'science museum', 'server room', 'shed', 'shoe shop', 'shopfront', 'shopping mall indoor', 'shower', 
    'ski resort', 'ski slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'soccer field', 'stable', 'stadium baseball', 'stadium football', 
    'stadium soccer', 'stage indoor', 'stage outdoor', 'staircase', 'storage room', 'street', 'subway station platform', 'supermarket', 
    'sushi bar', 'swamp', 'swimming hole', 'swimming pool indoor', 'swimming pool outdoor', 'synagogue outdoor', 'television room', 
    'television studio', 'temple asia', 'throne room', 'ticket booth', 'topiary garden', 'tower', 'toyshop', 'train interior', 
    'train station platform', 'tree farm', 'tree house', 'trench', 'tundra', 'underwater ocean deep', 'utility room', 'valley', 
    'vegetable garden', 'veterinarians office', 'viaduct', 'village', 'vineyard', 'volcano', 'volleyball court outdoor', 'waiting room', 
    'water park', 'water tower', 'waterfall', 'watering hole', 'wave', 'wet bar', 'wheat field', 'wind farm', 'windmill', 'yard', 'youth hostel', 'zen garden'
]

data_dict, targets_dict = {}, {}

for split in splits:
    supp = 'train_standard' if splits == 'train' else 'val'
    info_file = np.array(pd.read_csv(f'{root}/places365_{supp}.txt'))

    if split == 'train':
        subset_folder = os.path.join(root, 'PLACES365_subset_train')
        data = []
        targets = []
        classes = sorted(os.listdir(subset_folder))
        for i, folder in enumerate(classes):
            for file in os.listdir(os.path.join(subset_folder, folder)):
                data.append(os.path.join(subset_folder, folder, file))
                targets.append(i)
    else:
        data, targets = [], []                
        for x in info_file:
            path, target = x[0].split(' ')
            target = int(target)
            path = os.path.join(root, 'val', path)
            data.append(path)
            targets.append(target)          
    data_dict[split] = np.array(data)
    target_dict[split] = np.array(targets)

PLACES365_infodicts = {}
for split in splits:
    PLACES365_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        spl_val = 'PLACES365_subset_train/' if split == 'train' else ''
        ref_path = f'PLACES365/{spl_val}{classinfo}'
        PLACES365_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.oxford_pets.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(PLACES365_infodicts['train'], open(f'{infd_path}/PLACES365_train.json', 'w'), indent=4)
json.dump(PLACES365_infodicts['test'], open(f'{infd_path}/PLACES365_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/PLACES365_classnames.json', 'w'), indent=4)



#%% ##########################################################    PLANTVILLAGE
import data_lib.plantvillage
root = './data/PLANTVILLAGE'
splits = ['train', 'test']

data_dict = {}
target_dict = {}
data = []
targets = []
classes = sorted(os.listdir(root))
ordered_class_names = [
    'apple leaf with apple scab', 'apple leaf with black rot', 'apple leaf with cedar apple rust', 'healthy apple leaf', 
    'healthy blueberry leaf', 'cherry leaf with powdery mildew', 'healthy cherry leaf', 
    'corn leaf with cercospora leaf spot gray leaf spot', 'corn leaf with common rust', 'corn leaf with northern leaf blight', 
    'healthy corn leaf', 'grape leaf with black rot', 'grape leaf with esca (black measles)', 
    'grape leaf with leaf blight (isariopsis leaf spot)', 'healthy grape leaf', 'orange leaf with haunglongbing (citrus greening)', 
    'peach leaf with bacterial spot', 'healthy peach leaf', 'bell pepper leaf with bacterial spot', 'healthy bell pepper leaf', 
    'potato leaf with early blight', 'potato leaf with late blight', 'healthy potato leaf', 'healthy raspberry leaf', 
    'healthy soybean leaf', 'squash leaf with powdery mildew', 'strawberry leaf with leaf scorch', 'healthy strawberry leaf', 
    'tomato leaf with bacterial spot', 'tomato leaf with early blight', 'tomato leaf with late blight', 'tomato leaf with leaf mold', 
    'tomato leaf with septoria leaf spot', 'tomato leaf with spider mites two-spotted spider mite', 'tomato leaf with target spot', 
    'tomato leaf with tomato yellow leaf curl virus', 'tomato leaf with tomato mosaic virus', 'healthy tomato leaf'
]
for i, folder in enumerate(classes):
    for file in os.listdir(os.path.join(root, folder)):
        data.append(os.path.join(root, folder, file))
        targets.append(i)
test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))
        
for split in splits:    
    subset = train_idcs if split == 'train' else test_idcs
    data_dict[split] = np.array([data[i] for i in subset])
    target_dict[split] = np.array([targets[i] for i in subset])

caption_dict = pkl.load(open('./data/dataset_captions/PLANTVILLAGE_captions.pkl', 'rb'))

PLANTVILLAGE_infodicts = {}

for split in splits:
    PLANTVILLAGE_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'PLANTVILLAGE/{classinfo}'
        PLANTVILLAGE_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.plantvillage.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(PLANTVILLAGE_infodicts['train'], open(f'{infd_path}/PLANTVILLAGE_train.json', 'w'), indent=4)
json.dump(PLANTVILLAGE_infodicts['test'], open(f'{infd_path}/PLANTVILLAGE_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/PLANTVILLAGE_classnames.json', 'w'), indent=4)



#%% ##########################################################    QUILT
import data_lib.quilt
root = './data/QUILT'
splits = ['train', 'test']

#INFO: Download from Zonodo-link, follow https://quilt1m.github.io/

metadata = pd.read_csv('~/Downloads/quilt1m/quilt_1M_lookup.csv')
metadata = metadata.drop_duplicates(subset=['image_path'])

subsets = np.array(metadata['subset'])
splits = np.array(metadata['split'])
idcs = np.where(np.logical_and(subsets == 'quilt', splits == 'train'))[0]

np.random.seed(0)
idcs = sorted(np.random.choice(idcs, 120000, replace=False))

image_paths = np.array(metadata['image_path'])[idcs]
captions = np.array(metadata['caption'])[idcs]
pathologies = np.array(metadata['pathology'])[idcs]
corrected_texts = np.array(metadata['corrected_text'])[idcs]

classnames = []
extra_classnames = []
idcs = []
unique_classnames = []
classes_to_remove = [
    'Digital pathology and Pathology consultation',
    'No information provided and Text not in English',
    'Unclassified and Unclassified',
    'Unknown',
    'Not enough information to classify',
    'N/A',
    'Hepatic',
    'Allergy and Immunology and Head and Neck',
    'Anatomic pathology and Hematopathology', 'Bone and Orthopedic',
    'Breast pathology and Ophthalmic',
    'Breast pathology and Pulmonary', 'Cardiac and Cytopathology',
    'Cardiac and Infectious Disease', 'Colorectal and Cytopathology',
    'Cytopathology and Hepatobiliary', 'Cytopathology and Pediatric',
    'Dental Pathology and Metallurgy', 'Dermatopathology',
    'Dermatopathology and Soft tissue', 'Endocrine and Hepatic',
    'Endocrine and Hepatopathology', 'Endocrine and Pediatric',
    'Genetic and Ophthalmic', 'Gynecologic and Head and Neck',
    'Gynecologic and Infectious Disease',
    'Head and Neck and Musculoskeletal',
    'Hematopathology and Liver pathology',
    'Hematopathology and Rheumatology', 'Pulmonary and Respiratory'     
]

for i, pathology in tqdm.tqdm(enumerate(pathologies), total=len(pathologies)):
    try:
        pathos = sorted(eval(pathology))
    except:
        continue
    if len(pathos) > 1:
        classname = f'{pathos[0]} and {pathos[1]}'
        extra_classname = pathos[1]
    else:
        classname = pathos[0]
        extra_classname = None
    
    if classname not in classes_to_remove:
        classnames.append(classname)
        extra_classnames.append(extra_classname)
        idcs.append(i)        


idcs = np.array(idcs)
image_paths = image_paths[idcs]
captions = captions[idcs]
ordered_class_names = sorted(np.unique(classnames))
cls2tar = {key: i for i, key in enumerate(ordered_class_names)}
targets = np.array([cls2tar[cn] for cn in classnames])
captions = np.array([cap[:180] for cap in captions])
captions = np.array([cap[:-1] if cap[-1] == '.' else cap for cap in captions])

base_variants = [
    "f'A histopathology image of {classname}. {caption}.'",
    "f'{caption} in this histo-image of {classname}.'",
    "f'{classname} pathologies, where: {caption}.'",
    "f'A histopathology photo of {classname} ({caption}).'",
    "f'A photo of {classname} pathologies; {caption}.'",
    "f'An image of {classname} pathologies. One can see that: {caption}.'",    
    "f'{caption}.'",
    "f'{caption}.'",
    "f'{caption}.'",
]

np.random.seed(0)
final_captions = []
for caption, classname in tqdm.tqdm(zip(captions, classnames), total=len(captions)):
    cap_var = np.random.choice(base_variants)
    final_captions.append(eval(cap_var))
captions = np.array(final_captions)

test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))

classnames = np.array(classnames)
QUILT_infodicts = {}
for split in ['train', 'test']:
    QUILT_infodicts[split] = {}
    idcs = train_idcs if split == 'train' else test_idcs
    data_list = image_paths[idcs]
    target_list = targets[idcs]
    classname_list = classnames[idcs]
    caption_list = captions[idcs]

    for path, target, classname, caption in zip(data_list, target_list, classname_list, caption_list):
        classname = ordered_class_names[target]
        QUILT_infodicts[split][f'quilt1m_data/{path}'] = {
            'classname': classname,
            'default_caption': caption,
            'primer_caption': data_lib.quilt.PRIMER.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(QUILT_infodicts['train'], open(f'{infd_path}/QUILT_train.json', 'w'), indent=4)
json.dump(QUILT_infodicts['test'], open(f'{infd_path}/QUILT_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/QUILT_classnames.json', 'w'), indent=4)



#%% ##########################################################    RESISC45
import data_lib.resisc45
root = './data/RESISC45'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

ordered_class_names = [
    "airplane", "airport", "baseball diamond", "basketball court", "beach", "bridge", "chaparral", "church", "circular farmland", "cloud", "commercial area", 
    "dense residential", "desert", "forest", "freeway", "golf course", "ground track field", "harbor", "industrial area", "intersection", "island",
    "lake", "meadow", "medium residential", "mobile home park", "mountain", "overpass", "palace", "parking lot", "railway", "railway station",
    "rectangular farmland","river","roundabout","runway","sea ice","ship","snowberg","sparse residential", "stadium", "storage tank", "tennis court", 
    "terrace", "thermal power station", "wetland",
]

caption_dict = pkl.load(open('./data/dataset_captions/RESISC45_captions.pkl', 'rb'))

for split in splits:
    if split == 'train':
        files = sorted(list(pd.read_csv(os.path.join(root, 'resisc45-train.txt'), delimiter=' ', header=None)[0]))
    else:
        files = sorted(list(pd.read_csv(os.path.join(root, 'resisc45-test.txt'), delimiter=' ', header=None)[0]))        

    classes = ['_'.join(x.split('_')[:-1]) for x in files]
    class_to_idx = {classname: idx for idx, classname in enumerate(sorted(np.unique(classes)))}

    data_dict[split] = np.array([os.path.join(root, 'NWPU-RESISC45', classname, file) for file, classname in zip(files, classes)])
    target_dict[split] = np.array([class_to_idx[classname] for classname in classes])

RESISC45_infodicts = {}

for split in splits:
    RESISC45_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'RESISC45/NWPU-RESISC45/{classinfo}'
        RESISC45_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.resisc45.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(RESISC45_infodicts['train'], open(f'{infd_path}/RESISC45_train.json', 'w'), indent=4)
json.dump(RESISC45_infodicts['test'], open(f'{infd_path}/RESISC45_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/RESISC45_classnames.json', 'w'), indent=4)


#%% ##########################################################    SHAPES3D
import h5py
import data_lib.shapes3d
import pandas as pd
import itertools as it
import tqdm

root = './data/SHAPES3D'
splits = ['train', 'test']

ordered_class_names = [
    'red cube', 'orange cube', 'yellow cube', 'green cube', 'blue cube', 'pink cube',
    'red cylinder', 'orange cylinder', 'yellow cylinder', 'green cylinder', 'blue cylinder', 'pink cylinder',
    'red sphere', 'orange sphere', 'yellow sphere', 'green sphere', 'blue sphere', 'pink sphere',
    'red pill', 'orange pill', 'yellow pill', 'green pill', 'blue pill', 'pink pill',
]

with h5py.File(f'{root}/shapes3d.h5', 'r') as shapes3d_data:
    imgs = shapes3d_data['images'][()]
    lat_values = shapes3d_data['labels'][()]

num_latents = len(lat_values)
train_indices = list(np.random.choice(num_latents, 75000, replace=False))
test_indices = list(np.random.choice(list(set(range(num_latents)) - set(train_indices)), 25000, replace=False))

lat_names = ('floorCol', 'wallCol', 'objCol', 'objSize', 'objType', 'objAzimuth')
lat_sizes = np.array([10, 10, 10, 8, 4, 15])

floorCol = lat_values[:, 0]
wallCol = lat_values[:, 1]
objCol = lat_values[:, 2]
objAzimuth = lat_values[:, 5]

floorCol_conv = {
    0: ('red', 0), 0.1: ('orange', 1), 0.2: ('yellow', 2), 0.3: ('green', 3), 0.4: ('green', 3), 0.5: ('blue', 4), 0.6: ('blue', 4), 0.7: ('blue', 4), 0.8: ('pink', 5), 0.9: ('pink', 5)
}
wallCol_conv = {
    0: ('red', 0), 0.1: ('orange', 1), 0.2: ('yellow', 2), 0.3: ('green', 3), 0.4: ('green', 3), 0.5: ('blue', 4), 0.6: ('blue', 4), 0.7: ('blue', 4), 0.8: ('pink', 5), 0.9: ('pink', 5)
}
objCol_conv = {
    0: ('red', 0), 0.1: ('orange', 1), 0.2: ('yellow', 2), 0.3: ('green', 3), 0.4: ('green', 3), 0.5: ('blue', 4), 0.6: ('blue', 4), 0.7: ('blue', 4), 0.8: ('pink', 5), 0.9: ('pink', 5)
}
objSize_conv = {
    0.75: 'small', 0.82142857: 'small', 0.89285714: 'smaller', 0.96428571: 'normal-sized', 1.03571429: 'normal-sized', 1.10714286: 'larger', 1.17857143: 'large', 1.25: 'large'
}
objSize_conv = {np.round(key, 2): item for key, item in objSize_conv.items()}
objType_conv = {
    0: ('cube', 0), 1: ('cylinder', 1), 2: ('sphere', 2), 3: ('pill', 3)
}
objAzimuth_conv = {np.round(val, 2): '{0:2.2f} degree'.format(val, 2) for val in sorted(np.unique(objAzimuth))}
    
label_conv = {
    0: {key: item[1] for key, item in floorCol_conv.items()},
    1: {key: item[1] for key, item in wallCol_conv.items()},
    2: {key: item[1] for key, item in objCol_conv.items()},
    4: {key: item[1] for key, item in objType_conv.items()},
}

phrases = [
    'f"A photo of a synthetic {objsize}, {objcol} {objtype} on a {floorcol} floor and a {wallcol} background from a {azimuth} viewing angle."',
    'f"A {objcol} {objsize} {objtype} positioned on a {floorcol} surface against a {wallcol} backdrop, photographed from a {azimuth} angle."',
    'f"Captured from a {azimuth} angle: a {objcol}, {objsize} {objtype} against a {wallcol} wall on a {floorcol} flooring."',
    'f"An image showing a {objcol} {objtype} of {objsize} size, set against a {wallcol} wall and {floorcol} ground, taken from a {azimuth} viewpoint."',
    'f"From a {azimuth} perspective, this photo features a {objsize}, {objcol} {objtype} with a {floorcol} floor and {wallcol} background."',
    'f"A {objcol} {objtype} ({objsize} size), displayed on {floorcol} flooring with a {wallcol} background, viewed from a {azimuth} angle."',
    'f"Featuring a {objtype} in {objcol}, on a {floorcol}-colored floor and a {wallcol} background, from a {azimuth} viewing perspective."',
    'f"Depicting a {objsize}, {objcol} {objtype} staged on a {floorcol} base with a {wallcol} hue in the backdrop, captured from a {azimuth} direction."',
    'f"A {objcol} {objtype} set against a {wallcol} background, captured beautifully."',
    'f"An image featuring a {objsize} {objtype} on a {floorcol} surface."',
    'f"A {objtype} displayed prominently against a contrasting {wallcol} backdrop."',
    'f"A detailed shot of a {objtype}, highlighted by its {objcol} color."',
    'f"The elegance of a {objtype} captured from a unique angle."',
    'f"A {objsize} {objtype}, standing out on a simplistic background."',
    'f"A synthetic {objtype} in a synthetic environment."',
    'f"A {objtype} of {objsize} size, standing out with its vibrant {objcol}."',
    'f"From the {azimuth} direction, the {objcol} {objtype} makes a striking impression against a simple backdrop."',
    'f"The {azimuth} perspective brings out the vivid {objcol} color of the {objtype}, making it the focal point."'
]

muls = [1, 6, 36, 216]
targets = []
captions = []
t2c = {}
np.random.seed(0)
for lat in tqdm.tqdm(lat_values):
    lat = np.round(lat, 2)
    floorcol = floorCol_conv[lat[0]][0]
    wallcol = wallCol_conv[lat[1]][0]
    objcol = objCol_conv[lat[2]][0]
    objsize = objSize_conv[lat[3]]
    objtype = objType_conv[lat[4]][0]
    azimuth = objAzimuth_conv[lat[5]]
    cap_str = np.random.choice(phrases)
    target = 0
    for k, i in enumerate([0, 1, 2, 4]):
        target += (label_conv[i][float(lat[i])]) * muls[k]
    t2c[target] = f'{objcol} {objtype} on a {floorcol} floor with a {wallcol} wall'
    targets.append(target)
    captions.append(eval(cap_str))
targets = np.array(targets)    
captions = np.array(captions)


ordered_class_names = [t2c[i] for i in range(len(t2c))]
lat_names = ('floorCol', 'wallCol', 'objCol', 'objSize', 'objType', 'objAzimuth')



all_imgs = {'train': train_indices, 'test': test_indices}
all_targets = {'train': targets[train_indices], 'test': targets[test_indices]}
all_captions = {'train': captions[train_indices], 'test': captions[test_indices]}


# import matplotlib.pyplot as plt
# idcs = np.random.choice(len(all_imgs['train']), 10, replace=False)
# for i in idcs:
#     f, ax = plt.subplots()
#     img = all_imgs['train'][i]
#     target = all_targets['train'][i]
#     label = ordered_class_names[target]
#     caption = all_captions['train'][i]
#     ax.imshow(img)
#     ax.set_title(f'{target} | {label} | {caption}')
#     plt.show()

shapes3d_infodicts = {}
for split in splits:
    sub_data = all_imgs[split]
    sub_targets = all_targets[split]
    sub_captions = all_captions[split]
    
    shapes3d_infodicts[split] = {}
    
    for i, (index, target, caption) in enumerate(zip(sub_data, sub_targets, sub_captions)):
        classname = ordered_class_names[target]
        path = f'shapes3d-images-{split}/{index}.png'
        shapes3d_infodicts[split][path] = {
            'classname': classname,
            'default_caption': caption,
            'primer_caption': data_lib.shapes3d.PRIMER.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(shapes3d_infodicts['train'], open(f'{infd_path}/SHAPES3D_train.json', 'w'), indent=4)
json.dump(shapes3d_infodicts['test'], open(f'{infd_path}/SHAPES3D_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/SHAPES3D_classnames.json', 'w'), indent=4)



#%% ##########################################################    SNAKE_CLEF
import data_lib.snake_clef
import pandas as pd
import itertools as it
import tqdm

root = './data/SNAKECLEF'
splits = ['train', 'test']

train_metadata = pd.read_csv(f'{root}/train.csv')
val_metadata = pd.read_csv(f'{root}/val.csv')

train_classnames = np.array(train_metadata['binomial_name'])
val_classnames = np.array(val_metadata['binomial_name'])
train_targets = np.array(train_metadata['class_id'])
train_imagepaths = np.array(train_metadata['image_path'])
val_targets = np.array(val_metadata['class_id'])
val_imagepaths = np.array(val_metadata['image_path'])

a = set(sorted(np.unique(train_classnames)))
b = set(sorted(np.unique(val_classnames)))
final_classnames = sorted(list(a & b))
cls2tar = {}
for i, fincls in enumerate(final_classnames):
    cls2tar[fincls] = i

train_check = np.array([x in final_classnames for x in tqdm.tqdm(train_classnames)])
val_check = np.array([x in final_classnames for x in tqdm.tqdm(val_classnames)])

train_classnames = train_classnames[train_check]
val_classnames = val_classnames[val_check]

train_targets = np.array([cls2tar[clsn] for clsn in train_classnames])
val_targets = np.array([cls2tar[clsn] for clsn in val_classnames])
train_imagepaths = train_imagepaths[train_check]
val_imagepaths = val_imagepaths[val_check]

train_imagepaths = 'SnakeCLEF2023-medium_size/' + train_imagepaths
val_imagepaths = 'SnakeCLEF2023-medium_size/' + val_imagepaths


SNAKECLEF_infodicts = {}
for split in splits:
    sub_imagepaths = train_imagepaths if split == 'train' else val_imagepaths
    sub_targets = train_targets if split == 'train' else val_targets
    sub_classnames = train_classnames if split == 'train' else val_classnames
    
    SNAKECLEF_infodicts[split] = {}
    
    for i, (path, target, classname) in enumerate(zip(sub_imagepaths, sub_targets, sub_classnames)):
        if os.path.exists(f'{root}/{path}'):
            SNAKECLEF_infodicts[split][path] = {
                'classname': classname,
                'default_caption': None,
                'primer_caption': data_lib.snake_clef.PRIMER.format(classname),
                'synthetic_caption': None,
                'synthetic_merged_caption': None,
                'target': int(target)
            }

ordered_class_names = final_classnames
json.dump(SNAKECLEF_infodicts['train'], open(f'{infd_path}/SNAKECLEF_train.json', 'w'), indent=4)
json.dump(SNAKECLEF_infodicts['test'], open(f'{infd_path}/SNAKECLEF_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/SNAKECLEF_classnames.json', 'w'), indent=4)



#%% ##########################################################    RETINOPATHY
import data_lib.retinopathy

root = './data/RETINOPATHY'
splits = ['train', 'test']

base_folder = os.path.join(root, 'data')

caption_dict = pkl.load(open('./data/dataset_captions/RETINOPATHY_captions.pkl', 'rb'))

ordered_class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
conv = {key: i for i, key in enumerate(ordered_class_names)}
data = []
targets = []
for folder in ordered_class_names:
    for file in os.listdir(os.path.join(root, folder)):
        data.append(os.path.join(root, folder, file))
        targets.append(conv[folder])
data = np.array(data)
targets = np.array(targets)

test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))

data_dict = {'train': data[train_idcs], 'test': data[test_idcs]}
targets_dict = {'train': targets[train_idcs], 'test': targets[test_idcs]}

RETINOPATHY_infodicts = {}
ordered_class_names = ['an eye with no diabetic retinopathy', 'an eye with mild diabetic retinopathy', 'an eye with moderate diabetic retinopathy', 'an eye with severe diabetic retinopathy', 'an eye with proliferate diabetic retinopathy']
for split in splits:
    RETINOPATHY_infodicts[split] = {}
    data_list = data_dict[split]
    targets = targets_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'RETINOPATHY/{classinfo}'
        RETINOPATHY_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.retinopathy.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(RETINOPATHY_infodicts['train'], open(f'{infd_path}/RETINOPATHY_train.json', 'w'), indent=4)
json.dump(RETINOPATHY_infodicts['test'], open(f'{infd_path}/RETINOPATHY_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/RETINOPATHY_classnames.json', 'w'), indent=4)



#%% ##########################################################    STL10
import data_lib.stl10
root = './data/STL10'
splits = ['train', 'test']
datasets = {
    'train': torchvision.datasets.STL10(root, 'train', transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.STL10(root, 'test', transform=None, target_transform=None, download=True)
}

ordered_class_names = [x.split(' -')[0] for x in list(datasets['train'].classes)]
STL10_caption_dict = pkl.load(open('./data/dataset_captions/STL10_captions.pkl', 'rb'))

STL10_infodicts = {}

for split in splits:
    if split == 'train':
        data = list(range(len(datasets['train'].data)))
    else:
        data = list(range(len(datasets['train'].data), len(datasets['train'].data) + len(datasets['test'].data)))
    targets = np.array(datasets[split].labels)
    classnames = []
    for target in targets:
        classnames.append(ordered_class_names[target])

    STL10_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        sub = 0 if split == 'train' else len(datasets['train'].data)
        key_path = f'stl10-images-{split}/{target}-{path-sub}.png'            
        STL10_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.stl10.PRIMER.format(classname),
            'synthetic_caption': STL10_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': STL10_caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }
    STL10_infodicts[split] = STL10_infodict

json.dump(STL10_infodicts['train'], open(f'{infd_path}/STL10_train.json', 'w'), indent=4)
json.dump(STL10_infodicts['test'], open(f'{infd_path}/STL10_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/STL10_classnames.json', 'w'), indent=4)


#%% ##########################################################    SUN397
import data_lib.sun397
root = './data/SUN397'
splits = ['train', 'test']

data_dict = {}
target_dict = {}

for split in splits:    
    data = []
    targets = []
    classes = sorted(os.listdir(os.path.join(root, split)))
    for i, folder in enumerate(classes):
        for file in os.listdir(os.path.join(root, split, folder)):
            data.append(os.path.join(root, split, folder, file))
            targets.append(i)
    data_dict[split] = np.array(data)
    target_dict[split] = np.array(targets)

caption_dict = pkl.load(open('./data/dataset_captions/SUN397_captions.pkl', 'rb'))

ordered_class_names = [
    'abbey', 'airplane cabin', 'airport terminal', 'alley', 'amphitheater', 'amusement arcade', 'amusement park', 'anechoic chamber', 'apartment building outdoor', 'apse indoor', 'aquarium', 'aqueduct', 'arch', 'archive', 'arrival gate outdoor', 'art gallery', 'art school', 'art studio', 'assembly line', 'athletic field outdoor', 'atrium public', 'attic', 'auditorium', 'auto factory', 'badlands', 'badminton court indoor', 'baggage claim', 'bakery shop', 'balcony exterior', 'balcony interior', 'ball pit', 'ballroom', 'bamboo forest', 'banquet hall', 'bar', 'barn', 'barndoor', 'baseball field', 'basement', 'basilica', 'basketball court outdoor', 'bathroom', 'batters box', 'bayou', 'bazaar indoor',
    'bazaar outdoor', 'beach', 'beauty salon', 'bedroom', 'berth', 'biology laboratory', 'bistro indoor', 'boardwalk', 'boat deck', 'boathouse', 'bookstore', 'booth indoor', 'botanical garden', 'bow window indoor', 'bow window outdoor', 'bowling alley', 'boxing ring', 'brewery indoor', 'bridge', 'building facade', 'bullring', 'burial chamber', 'bus interior', 'butchers shop', 'butte', 'cabin outdoor', 'cafeteria', 'campsite', 'campus', 'canal natural', 'canal urban', 'candy store', 'canyon', 'car interior backseat', 'car interior frontseat', 'carrousel', 'casino indoor', 'castle', 'catacomb', 'cathedral indoor', 'cathedral outdoor', 'cavern indoor', 'cemetery', 'chalet', 'cheese factory', 'chemistry lab', 'chicken coop indoor', 'chicken coop outdoor', 'childs room', 'church indoor', 'church outdoor', 'classroom', 'clean room', 'cliff', 'cloister indoor', 'closet', 'clothing store', 'coast', 'cockpit', 'coffee shop', 'computer room', 'conference center', 'conference room', 'construction site', 'control room', 'control tower outdoor', 'corn field', 'corral', 'corridor', 'cottage garden', 'courthouse',
    'courtroom', 'courtyard', 'covered bridge exterior', 'creek', 'crevasse', 'crosswalk', 'cubicle office', 'dam', 'delicatessen', 'dentists office', 'desert sand', 'desert vegetation', 'diner indoor', 'diner outdoor', 'dinette home', 'dinette vehicle', 'dining car', 'dining room', 'discotheque', 'dock', 'doorway outdoor', 'dorm room', 'driveway', 'driving range outdoor', 'drugstore', 'electrical substation', 'elevator door', 'elevator interior', 'elevator shaft', 'engine room', 'escalator indoor', 'excavation', 'factory indoor', 'fairway', 'fastfood restaurant', 'field cultivated', 'field wild', 'fire escape', 'fire station', 'firing range indoor', 'fishpond', 'florist shop indoor', 'food court', 'forest broadleaf', 'forest needleleaf', 'forest path', 'forest road',
    'formal garden', 'fountain', 'galley', 'game room', 'garage indoor', 'garbage dump', 'gas station', 'gazebo exterior', 'general store indoor', 'general store outdoor', 'gift shop', 'golf course', 'greenhouse indoor', 'greenhouse outdoor', 'gymnasium indoor', 'hangar indoor', 'hangar outdoor', 'harbor', 'hayfield', 'heliport', 'herb garden', 'highway', 'hill', 'home office', 'hospital', 'hospital room', 'hot spring', 'hot tub outdoor', 'hotel outdoor', 'hotel room', 'house', 'hunting lodge outdoor', 'ice cream parlor', 'ice floe', 'ice shelf', 'ice skating rink indoor', 'ice skating rink outdoor', 'iceberg', 'igloo', 'industrial area', 'inn outdoor', 'islet', 'jacuzzi indoor', 'jail cell',
    'jail indoor', 'jewelry shop', 'kasbah', 'kennel indoor', 'kennel outdoor', 'kindergarden classroom', 'kitchen', 'kitchenette', 'labyrinth outdoor', 'lake natural', 'landfill', 'landing deck', 'laundromat', 'lecture room', 'library indoor', 'library outdoor', 'lido deck outdoor', 'lift bridge', 'lighthouse', 'limousine interior', 'living room', 'lobby', 'lock chamber', 'locker room', 'mansion', 'manufactured home', 'market indoor', 'market outdoor', 'marsh', 'martial arts gym', 'mausoleum', 'medina', 'moat water', 'monastery outdoor', 'mosque indoor', 'mosque outdoor', 'motel', 'mountain', 'mountain snowy', 'movie theater indoor', 'museum indoor',
    'music store', 'music studio', 'nuclear power plant outdoor', 'nursery', 'oast house', 'observatory outdoor', 'ocean', 'office', 'office building', 'oil refinery outdoor', 'oilrig', 'operating room', 'orchard', 'outhouse outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking garage indoor', 'parking garage outdoor', 'parking lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone booth', 'physics laboratory', 'picnic area', 'pilothouse indoor', 'planetarium outdoor', 'playground', 'playroom', 'plaza', 'podium indoor', 'podium outdoor', 'pond', 'poolroom establishment', 'poolroom home',
    'power plant outdoor', 'promenade deck', 'pub indoor', 'pulpit', 'putting green', 'racecourse', 'raceway', 'raft', 'railroad track', 'rainforest', 'reception', 'recreation room', 'residential neighborhood', 'restaurant', 'restaurant kitchen', 'restaurant patio', 'rice paddy', 'riding arena', 'river', 'rock arch', 'rope bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea cliff', 'server room', 'shed', 'shoe shop', 'shopfront', 'shopping mall indoor', 'shower', 'skatepark', 'ski lodge', 'ski resort', 'ski slope', 'sky', 'skyscraper', 'slum', 'snowfield',
    'squash court', 'stable', 'stadium baseball', 'stadium football', 'stage indoor', 'staircase', 'street', 'subway interior', 'subway station platform', 'supermarket', 'sushi bar', 'swamp', 'swimming pool indoor', 'swimming pool outdoor', 'synagogue indoor', 'synagogue outdoor', 'television studio', 'temple east asia', 'temple south asia', 'tennis court indoor', 'tennis court outdoor', 'tent outdoor', 'theater indoor procenium', 'theater indoor seats', 'thriftshop', 'throne room', 'ticket booth', 'toll plaza', 'topiary garden', 'tower', 'toyshop', 'track outdoor', 'train railway', 'train station platform', 'tree farm', 'tree house', 'trench', 'underwater coral reef', 'utility room',
    'valley', 'van interior', 'vegetable garden', 'veranda', 'veterinarians office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball court indoor', 'volleyball court outdoor', 'waiting room', 'warehouse indoor', 'water tower', 'waterfall block', 'waterfall fan', 'waterfall plunge', 'watering hole', 'wave', 'wet bar', 'wheat field', 'wind farm', 'windmill', 'wine cellar barrel storage', 'wine cellar bottle storage', 'wrestling ring indoor', 'yard', 'youth hostel'
]

SUN397_infodicts = {}

for split in splits:
    SUN397_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'SUN397/SUN397_{split}_224/{classinfo}'
        SUN397_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.sun397.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(SUN397_infodicts['train'], open(f'{infd_path}/SUN397_train.json', 'w'), indent=4)
json.dump(SUN397_infodicts['test'], open(f'{infd_path}/SUN397_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/SUN397_classnames.json', 'w'), indent=4)


#%% ##########################################################    SVHN
import data_lib.svhn
root = './data/SVHN'
splits = ['train', 'test']
datasets = {
    'train': torchvision.datasets.SVHN(root, 'train', transform=None, target_transform=None, download=True),
    'test': torchvision.datasets.SVHN(root, 'test', transform=None, target_transform=None, download=True)
}

ordered_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

SVHN_caption_dict = pkl.load(open('./data/dataset_captions/SVHN_captions.pkl', 'rb'))

SVHN_infodicts = {}

for split in splits:
    data = list(range(len(datasets[split].data)))
    targets = np.array(datasets[split].labels)
    classnames = []
    for target in targets:
        classnames.append(ordered_class_names[target])

    SVHN_infodict = {}
    for path, classname, target in zip(data, classnames, targets):
        ref_path = path
        key_path = f'svhn-images-{split}/{target}-{path}.png'        
        SVHN_infodict[key_path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.svhn.PRIMER.format(classname),
            'synthetic_caption': SVHN_caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': SVHN_caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }
    SVHN_infodicts[split] = SVHN_infodict

json.dump(SVHN_infodicts['train'], open(f'{infd_path}/SVHN_train.json', 'w'), indent=4)
json.dump(SVHN_infodicts['test'], open(f'{infd_path}/SVHN_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/SVHN_classnames.json', 'w'), indent=4)


#%% ##########################################################    VEG200
import shutil
import zipfile
import data_lib.veg200
root = './data/VEG200'
splits = ['train', 'test']
base_path = os.path.join(root, 'processed', 'vegfru_list')
# if 'archive.zip' in os.listdir(root):
#     archive_file = os.path.join(root, 'archive.zip')
# else:
#     archive_file = os.path.join(veg200_root, 'archive.zip')
    
# with zipfile.ZipFile(archive_file,"r") as zip_ref:
#     zip_ref.extractall(os.path.join(root, 'processed'))

# fru92_veg_path = os.path.join(root, 'processed', 'veg200_images')
# shutil.move(fru92_veg_path, os.path.join(veg200_root, 'processed'))
# shutil.copytree(base_path, os.path.join(veg200_root, 'processed', 'vegfru_list'))
            
files = {
    'train': pd.read_csv(os.path.join(base_path, f'vegfru_test.txt'), delimiter=' ', header=None),
    'test': pd.read_csv(os.path.join(base_path, f'vegfru_train.txt'), delimiter=' ', header=None)
}

file_dict = {split: files[split][0] for split in splits}
target_dict = {split: files[split][1] for split in splits}
data_dict = {split: [] for split in splits}
classnames = {}
for split in splits:
    targets =[]
    for i in tqdm.trange(len(file_dict[split])):
        if 'veg200_images' in file_dict[split][i]:
            data_dict[split].append(os.path.join(root, 'processed', file_dict[split][i]))
            target = target_dict[split][i]
            targets.append(target)
            classnames[target] = file_dict[split][i].split('/')[1].replace('_', ' ')
    min_target = min(targets)
    targets = [x - min_target for x in targets]      
    target_dict[split] = targets

caption_dict = pkl.load(open('./data/dataset_captions/VEG200_captions.pkl', 'rb'))

unique_targets = sorted(np.unique(target_dict[split]))
ordered_class_names = [classnames[target + min_target] for target in unique_targets]

VEG200_infodicts = {}
for split in splits:
    VEG200_infodicts[split] = {}
    data_list = data_dict[split]
    targets = target_dict[split]
    
    for path, target in zip(data_list, targets):
        path = str(path.replace(f'{root}/',''))
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'VEG200/VEG200_{split}_224/{classinfo}'
        VEG200_infodicts[split][path] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.veg200.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(VEG200_infodicts['train'], open(f'{infd_path}/VEG200_train.json', 'w'), indent=4)
json.dump(VEG200_infodicts['test'], open(f'{infd_path}/VEG200_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/VEG200_classnames.json', 'w'), indent=4)


#%% ##########################################################    ZAPPOS50K
import data_lib.zappos50k

root = './data/ZAPPOS50k'
splits = ['train', 'test']

base_folder = os.path.join(root, 'data')

caption_dict = pkl.load(open('./data/dataset_captions/ZAPPOS50k_captions.pkl', 'rb'))

classnames = sorted(os.listdir(base_folder))

data = []
targets = []
for i, folder in enumerate(classnames):
    for file in os.listdir(os.path.join(base_folder, folder)):
        data.append(os.path.join(base_folder, folder, file))
        targets.append(i)
data = np.array(data)
targets = np.array(targets)

a, b = np.unique(targets, return_counts=True)
ixs = np.where(b < 5)[0]

data = []
targets = []
ordered_class_names = []

count = 0
for i, folder in enumerate(classnames):
    if i not in ixs:        
        for file in os.listdir(os.path.join(base_folder, folder)):
            data.append(os.path.join(base_folder, folder, file))
            targets.append(count)
        ordered_class_names.append(folder)
        count += 1
data = np.array(data)
targets = np.array(targets)

test_idcs = range(0, len(targets), 5)
train_idcs = list(set(range(len(targets))) - set(test_idcs))

data_dict = {'train': data[train_idcs], 'test': data[test_idcs]}
targets_dict = {'train': targets[train_idcs], 'test': targets[test_idcs]}

ZAPPOS50k_infodicts = {}
ordered_class_names = [x.replace('_', ' ') for x in ordered_class_names]
for split in splits:
    ZAPPOS50k_infodicts[split] = {}
    data_list = data_dict[split]
    targets = targets_dict[split]
    
    for path, target in zip(data_list, targets):
        classname = ordered_class_names[target]
        classinfo = '/'.join(path.split('/')[-2:])
        ref_path = f'ZAPPOS50k/data/{classinfo}'
        ZAPPOS50k_infodicts[split][path.replace(f'{root}/','')] = {
            'classname': classname,
            'default_caption': None,
            'primer_caption': data_lib.zappos50k.PRIMER.format(classname),
            'synthetic_caption': caption_dict[ref_path]['synthetic_caption'],
            'synthetic_merged_caption': caption_dict[ref_path]['merged_caption'],
            'target': int(target)
        }

json.dump(ZAPPOS50k_infodicts['train'], open(f'{infd_path}/ZAPPOS50k_train.json', 'w'), indent=4)
json.dump(ZAPPOS50k_infodicts['test'], open(f'{infd_path}/ZAPPOS50k_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/ZAPPOS50k_classnames.json', 'w'), indent=4)





#%%




#%%

#%% ##########################################################    CLEVR
root = './data/CLEVR'
data_url = 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip'
# torchvision.datasets.utils.download_and_extract_archive(data_url, download_root=root)

metadata = json.load(open(f'{root}/CLEVR_v1.0/scenes/CLEVR_train_scenes.json', 'r'))
image_paths = [f'{root}/CLEVR_v1.0/images/train/{x}' for x in os.listdir(f'{root}/CLEVR_v1.0/images/train')]

summary = []
relationships = []
filenames = []
for i in range(len(metadata['scenes'])):
    objects = metadata['scenes'][i]['objects']
    relationships.append(metadata['scenes'][i]['relationships'])
    object_list = [[[x['shape']], x['color'], x['material'], x['size']] for x in objects]
    summary.append(object_list)
    filenames.append(metadata['scenes'][i]['image_filename'])

conversion = {
    1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'
    
}

#%% CREATE CUSTOM CAPTIONS
np.random.seed(0)

captions = []
itrange = range(len(metadata['scenes']))
string_vars = [
    '',
    'different objects, such as ',
    'multiple shapes, e.g. ',
]
prep_vars = [
    'A scene with ',
    'A photo of synthetic shapes: ',
    'Several synthetic objects; ',
    'A photo of ',
    'A synthetic scene of '
]
for i in tqdm.tqdm(itrange):
    x = summary[i]
    obj_types, obj_counts = np.unique(['_'.join(y[0]) for y in sorted(x)], return_counts=True)    
    objects = [a[0][0] for a in x]
    obj_idx = np.random.choice(len(obj_types))
    app = 's' if obj_counts[obj_idx] > 1 else ''

    # Describe only one object type.
    descriptors = [' '.join(np.random.choice(r[1:], len(r[1:]), replace=False)) for r in x if r[0][0] == obj_types[obj_idx]]
    descr_str = ', '.join('a {} one'.format(r) for r in descriptors[:4])
    if len(descriptors) > 4:
        descr_str = 'such as ' + descr_str
    string_var = np.random.choice(string_vars)
    prep_var = np.random.choice(prep_vars)
    base_str = f'{prep_var}{string_var}{conversion[obj_counts[obj_idx]]} {obj_types[obj_idx]}{app}; {descr_str}.'

    classname = ''
    idcs = np.arange(len(obj_types))
    np.random.shuffle(idcs)
    obj_types = obj_types[idcs]
    obj_counts = obj_counts[idcs]
    for k, (ai, bi) in enumerate(zip(obj_types, obj_counts)):
        if bi > 1:
            ai = ai + 's'
        classname += f'{conversion[bi]} {ai}'
        if k < len(obj_types) - 2:
            classname += ', '    
        if k == len(obj_types) - 2:
            classname += ' and '
    
    descriptors = [' '.join(np.random.choice(r[1:], len(r[1:]), replace=False)) for r in x]
    part = {
        'right': 'right of', 'left': 'left of', 'behind': 'further behind than', 'front': 'further in front than'        
    }
    
    # One relation.
    rel_type = np.random.choice(list(relationships[i].keys()))
    obj_idx = np.random.choice(np.where([len(r) > 0 for r in relationships[i][rel_type]])[0])
    sub_obj_idx = np.random.choice(relationships[i][rel_type][obj_idx])
    aug = f'; the {descriptors[sub_obj_idx]} {objects[sub_obj_idx]} is {part[rel_type]} the {descriptors[obj_idx]} {objects[obj_idx]}'
    adv_str = f'{prep_var}{classname}{aug}'
     
    # Two relations.
    rel_type = np.random.choice(list(relationships[i].keys()))
    obj_idx = np.random.choice(np.where([len(r) > 0 for r in relationships[i][rel_type]])[0])
    sub_obj_idx = np.random.choice(relationships[i][rel_type][obj_idx])
    aug2 = f'the {descriptors[sub_obj_idx]} {objects[sub_obj_idx]} is {part[rel_type]} the {descriptors[obj_idx]} {objects[obj_idx]}'
    adv_adv_str = f'{prep_var}{classname}{aug}. In addition, {aug2}.'

    # classname, with one example:
    
    rand_idx = np.random.choice(len(objects))
    ex_str = f'{prep_var}{classname}, e.g. a {descriptors[rand_idx]} {objects[rand_idx]}.'
    
    # Use only the classname:
    cls_str = f'{prep_var}{classname}.'

    captions.append([adv_adv_str, adv_str, base_str, ex_str, cls_str])

captions = np.array(captions)

classnames = []
idcs_to_keep = []
classes_to_ignore = [
    'eight cubes', 'eight cubes, one cylinder',
    'eight cubes, one sphere', 'eight cubes, two cylinders',
    'eight cubes, two spheres', 'eight cylinders, one sphere',
    'eight spheres', 'nine cubes', 'nine cubes, one cylinder',
    'nine cubes, one sphere', 'nine cylinders, one sphere',
    'one cube, eight cylinders', 'one cube, eight spheres',
    'one cube, seven spheres', 'one cylinder, eight spheres',
    'one cylinder, nine spheres', 'seven cubes', 'seven cylinders',
    'ten spheres', 'two cubes, eight cylinders',
    'two cubes, eight spheres', 'two cylinders, eight spheres'
]

class_dict = {}
paths = []
caps = []
for idx, x in enumerate(summary):
    a, b = np.unique(['_'.join(y[0]) for y in sorted(x)], return_counts=True)
    classname = ''
    for i, (ai, bi) in enumerate(zip(a, b)):
        if bi > 1:
            ai = ai + 's'
        classname += f'{conversion[bi]} {ai}'
        if i < len(a) - 1:
            classname += ', '
    if classname not in classes_to_ignore:
        if classname not in class_dict:
            class_dict[classname] = []
        paths.append(metadata['scenes'][idx]['image_filename'])
        classnames.append(classname)
        caps.append(captions[idx])
        idcs_to_keep.append(idx)
        
classnames = np.array(classnames)
caps = np.array(caps)
paths = np.array(paths)

sort_idx = np.argsort(classnames)
classnames = classnames[sort_idx]
caps = caps[sort_idx]
paths = paths[sort_idx]

test_idcs = range(0, len(classnames), 5)
train_idcs = list(set(range(len(classnames))) - set(test_idcs))

CLEVR_infodicts = {}
ordered_class_names = sorted(np.unique(classnames))
cls2tar = {c: i for i, c in enumerate(ordered_class_names)}
for split in ['train', 'test']:
    CLEVR_infodicts[split] = {}
    idcs = train_idcs if split == 'train' else test_idcs

    data_list = paths[idcs]
    clsns = classnames[idcs]
    targets = np.array([cls2tar[clsn] for clsn in clsns])
    caption_list = caps[idcs]
    
    for path, target, clsn, caption in zip(data_list, targets, classnames, caption_list):
        classname = clsn
        split_val = path.split('_')[1]
        path = f'CLEVR_v1.0/images/{split_val}/{path}'
        CLEVR_infodicts[split][path] = {
            'classname': classname,
            'default_caption': list(caption),
            'primer_caption': 'A photo of synthetic shapes, with {}.'.format(classname),
            'synthetic_caption': None,
            'synthetic_merged_caption': None,
            'target': int(target)
        }

json.dump(CLEVR_infodicts['train'], open(f'{infd_path}/CLEVR_train.json', 'w'), indent=4)
json.dump(CLEVR_infodicts['test'], open(f'{infd_path}/CLEVR_test.json', 'w'), indent=4)
json.dump(ordered_class_names, open(f'{infd_path}/CLEVR_classnames.json', 'w'), indent=4)


#%%

#%% #################### EXTRA STUFF






#%%
# from IPython import embed; embed()        
# means, stds = [], []
# import tqdm
# for d in tqdm.tqdm(data):
#     x = np.array(Image.open(d).convert('RGB')).reshape(-1, 3)
#     means.append(np.mean(x, axis=0))
#     stds.append(np.std(x, axis=0))
# means = np.vstack(means).mean(0) / 255.
# stds = np.vstack(stds).mean(0) / 255.

# import matplotlib.pyplot as plt
# paths = np.array(datasets['test'].data)
# targets = np.array(datasets['test'].targets)
# idcs = np.random.choice(len(paths), 10, replace=False)
# for i in idcs:
#     f, ax = plt.subplots(1)
#     ax.imshow(paths[i], cmap='Grays')
#     ax.set_title(f'{targets[i]}, {ordered_class_names[targets[i]]}')
#     plt.show()
    
# idcs = np.random.choice(len(data), 10, replace=False)
# for i in idcs:
#     img = Image.open(data[i])
#     caption = caption_data[i]
#     f, ax = plt.subplots(1)
#     ax.imshow(img)
#     ax.set_title(caption)
#     plt.show()