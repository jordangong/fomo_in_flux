import copy
import datetime
from typing import Union, List, Dict

import numpy as np
import omegaconf
import termcolor
import torch
import tqdm

class DataSubset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        subset_idcs: np.ndarray,
        is_mask: bool = True,
    ):
        self.dataset = dataset
        self.PARAMS = dataset.PARAMS
        if is_mask:
            self.subset_idcs = np.where(subset_idcs)[0]
        else:
            self.subset_idcs = subset_idcs

    def __getitem__(self, index: int):
        index = self.subset_idcs[index]
        return self.dataset[index]

    def __len__(self):
        return len(self.subset_idcs)



class AggregateDataSubset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: Dict[str,torch.utils.data.Dataset],
        subset_idcs: List[int],
        ds_for_subset_idcs: List[str],
        auxiliary_context: List[Dict] = None
    ):
        """Slice through multiple different datasets without restructuring.

        Args:
            datasets (Dict[str,torch.utils.data.Dataset]): Dict of dataset_name: respective dataset to subsample from.
            subset_idcs (List[int]): List of sample indices w.r.t. to each dataloader.
            ds_for_subset_idcs (List[str]): List of dataset handles for each subset idx.
            auxiliary_context (List[Dict]): Same length as subset_idcs. 
                For any entry in subset_idcs, auxiliary context will be appended to the output!
        """
        self.datasets = datasets
        assert_str = 'len(subset_idcs) needs to be len(ds_for_subset_idcs)!'
        assert len(subset_idcs) == len(ds_for_subset_idcs), assert_str
        self.subset_idcs = subset_idcs
        self.ds_for_subset_idcs = ds_for_subset_idcs
        self.auxiliary_context = auxiliary_context
        self.num_samples = len(self.subset_idcs)
        
    def __getitem__(self, index: int):
        index = index % len(self.subset_idcs)
        dataset_name = self.ds_for_subset_idcs[index]
        index = self.subset_idcs[index]
        item_dict = self.datasets[dataset_name][index]
        if self.auxiliary_context is not None:
            if isinstance(self.auxiliary_context[index], dict):
                if len(self.auxiliary_context[index]):
                    item_dict.update(self.auxiliary_context[index])
        return item_dict

    def __len__(self):
        return self.get_len()

    def set_len(self, new_len):
        self.num_samples = new_len
    
    def get_len(self):
        return self.num_samples


class EndOfStreamReachedException(Exception):
    pass


class EvalOnlyExperiment:
    def __init__(self, args, eval_dataset):
        self.args = args
        self.eval_dataset = eval_dataset

        self.evaluation_batch_size = self.args.experiment.evaluation.batch_size
        self.num_workers = self.args.experiment.dataset.num_workers

    def give_test_dataloader(self):
        collate_fn = None
        if hasattr(self.eval_dataset, "_collate_fn"):
            collate_fn = self.eval_dataset._collate_fn
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            pin_memory=True,
            drop_last=False,
            batch_size=self.evaluation_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )








##########################################################
class BaseExperiment:
    
    def __init__(self, args: omegaconf.DictConfig, device: torch.device):
        self.args = args
        self.device = device

        self.num_workers = self.args.experiment.dataset.num_workers
        self.task_batch_size = self.args.experiment.task.batch_size
        if self.args.experiment.task.update_pool_batch_size:
            self.task_batch_size = self.args.experiment.task.update_pool_batch_size
        self.task_buffer_batch_size = self.args.experiment.task.buffer_pool_batch_size
        self.evaluation_batch_size = self.args.experiment.evaluation.batch_size
                
    def finish_task(self):
        """If task has been trained on, this is called.
        """
        self.task += 1

    def collate_fn(self, batch, custom_collate=None):
        """Defines custom collation function if needed.

        Args:
            batch (_type_): _description_
            custom_collate (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if custom_collate is None:
            return torch.utils.data.default_collate(batch)

    def has_end_been_reached(self):
        """Query to test if last task has been reached.

        Raises:
            EndOfStreamReachedException: Will be caught in a controlled manner when raised during adaptation.
        """
        end_reached = self.task >= self.num_tasks
        if end_reached:
            self.task = self.num_tasks
            raise EndOfStreamReachedException

    def get_task_indices(self, targets: Union[List, np.ndarray], task: int) -> np.ndarray:
        raise NotImplementedError()
    
    def give_task_datasets(self, task: int = None):
        raise NotImplementedError()

    def make_dataloader(self, dset, train=True):
        raise NotImplementedError()
    
    def give_task_dataloaders(self, task: int = None):
        raise NotImplementedError()
        # return {"train": train_loader, "test": test_loader}

    def give_full_datasets(self):
        raise NotImplementedError()        
        # return {"train": self.train_dataset, "test": self.test_dataset}

    def give_full_dataloaders(self):
        raise NotImplementedError()
        # train_loader, test_loader = None, None
        # return {"train": None, "test": None}

    def give_all_test_dataloaders(self, task_idx: Union[int, list] = None):
        raise NotImplementedError()

    @property
    def checkpoint(self):
        raise NotImplementedError()

    def load_from_checkpoint(self, state_dict):
        raise NotImplementedError()







##########################################################################
class PredefinedSequenceExperiment(BaseExperiment):
    
    def __init__(
        self, 
        args: omegaconf.DictConfig,
        train_datasets: List[torch.utils.data.Dataset],
        test_datasets: List[torch.utils.data.Dataset],
        dataset_names: List[str],
        device: torch.device,
        task_sequence: List = None,
        **kwargs
    ):
        super().__init__(args, device)
        
        self.task = 0
        self.task_sequence = task_sequence
        self.dataset_names = dataset_names
               
        self.total_num_classes = len(np.unique([x['global_target'] for y in task_sequence for x in y]))
        
        self.train_datasets, self.test_datasets = {}, {}
        for i, dataset_name in enumerate(dataset_names):
            self.train_datasets[dataset_name] = train_datasets[i]
            self.test_datasets[dataset_name] = test_datasets[i]
        
        self.get_all_subsampling_info()
        
        exp_time = (
            str(datetime.datetime.now())
            .replace(" ", "_")
            .replace(":", "-")
            .replace(".", "-")
        )        
        self.name = f"fomoinflux_{len(self.dataset_names)}_{exp_time}"
        
    def finish_task(self):
        self.task += 1

    @property
    def global_task(self):
        """PredefinedTaskSequences do not separate between dataset-specific and ...?

        Returns:
            _type_: _description_
        """
        return self.task
    
    def has_end_been_reached(self):
        end_reached = self.task >= self.args.experiment.task.num
        if end_reached:
            self.task = self.num_tasks
            raise EndOfStreamReachedException
        
    def collate_fn(self, batch, custom_collate=None):
        # Alter this to incorporate custom batch collation.
        if custom_collate is not None:
            return custom_collate(batch)
        return torch.utils.data.default_collate(batch)

    def get_all_subsampling_info(self):
        num_tasks = len(self.task_sequence)
        self.all_train_aux, self.all_test_aux = {}, {}
        self.all_train_idcs, self.all_test_idcs = {}, {}
        for dataset_name in self.train_datasets.keys():
            self.all_train_idcs[dataset_name] = [None for _ in range(num_tasks)]
            self.all_test_idcs[dataset_name] = [None for _ in range(num_tasks)]
            self.all_train_aux[dataset_name] = [None for _ in range(num_tasks)]
            self.all_test_aux[dataset_name] = [None for _ in range(num_tasks)]
        
        for task in tqdm.tqdm(range(num_tasks), desc='Preparing streaming info...'):
            task_datasets = [x['dataset'] for x in self.task_sequence[task]]
            task_targets = [x['target'] for x in self.task_sequence[task]]
            
            # Regroup.
            
            ds_task_train_idcs = {dataset_name: [] for dataset_name in task_datasets}
            ds_task_test_idcs = {dataset_name: [] for dataset_name in task_datasets}
            for dataset_name, target in zip(task_datasets, task_targets):
                train_dataset = self.train_datasets[dataset_name]
                test_dataset = self.test_datasets[dataset_name]
                ds_task_train_idcs[dataset_name].append(np.where(np.array(train_dataset.targets) == target)[0])
                ds_task_test_idcs[dataset_name].append(np.where(np.array(test_dataset.targets) == target)[0])
            for dataset_name in np.unique(task_datasets):
                self.all_train_idcs[dataset_name][task] = np.concatenate(ds_task_train_idcs[dataset_name])
                self.all_test_idcs[dataset_name][task] = np.concatenate(ds_task_test_idcs[dataset_name])
                  
    def get_task_indices(self, dataset_name: str, task: int=None):
        if task is None:
            task = self.task
        test_dataset = self.test_datasets[dataset_name]
        task_indices = np.zeros(len(test_dataset))
        task_classes = [x['target'] for x in self.task_sequence[task] if x['dataset'] == dataset_name]
        for task_class in task_classes:
            task_indices = np.logical_or(task_indices, test_dataset.targets == task_class)
        return np.where(task_indices)[0]
    
    def give_task_dataset_names(self, task: int=None):
        if task is None:
            task = self.task
        task_dataset_names = []
        for x in self.task_sequence[task]:
            if x['dataset'] not in task_dataset_names:
                task_dataset_names.append(x['dataset'])
        return task_dataset_names
    
    def give_task_datasets(self, task: int = None):
        # If we call give_task_datasets(), it will choose
        # the task based on current self.task.
        set_to_current = task is None
        if set_to_current:
            task = self.task

        self.has_end_been_reached()

        train_idcs, test_idcs = [], []
        train_ds, test_ds = [], []
        for dataset_name in self.dataset_names:
            ds_train_idcs = self.all_train_idcs[dataset_name][task]
            ds_test_idcs = self.all_test_idcs[dataset_name][task]
            if ds_train_idcs is not None:
                train_idcs.extend(list(ds_train_idcs))
                test_idcs.extend(list(ds_test_idcs))
                train_ds.extend(list(np.repeat(dataset_name, len(ds_train_idcs))))
                test_ds.extend(list(np.repeat(dataset_name, len(ds_test_idcs))))
        train_auxiliary_context = None                                    
        test_auxiliary_context = None
        
        task_train_dataset = AggregateDataSubset(
            self.train_datasets, train_idcs, train_ds, train_auxiliary_context)
        task_test_dataset = AggregateDataSubset(
            self.test_datasets, test_idcs, test_ds, test_auxiliary_context)
        
        # Also provide a buffer dataloader if needed.
        task_train_buffer_dataset = None
        if self.task_buffer_batch_size and task > 0:
            buffer_idcs, buffer_ds = [], []
            for buffer_task in range(task):
                for dataset_name in self.dataset_names:
                    buffer_sub_idcs = self.all_train_idcs[dataset_name][buffer_task]
                    if buffer_sub_idcs is not None:
                        buffer_sub_ds = np.repeat(dataset_name, len(buffer_sub_idcs))
                        buffer_idcs.extend(list(buffer_sub_idcs))
                        buffer_ds.extend(list(buffer_sub_ds))                    
            task_train_buffer_dataset = AggregateDataSubset(
                self.train_datasets, buffer_idcs, buffer_ds, None)
                    
        self.task_num_samples_train = len(task_train_dataset)
        self.task_num_samples_test = len(task_test_dataset)
        self.task_num_samples_buffer = 0 if not task_train_buffer_dataset else len(task_train_buffer_dataset)
        
        if set_to_current:
            self.current_task_train_dataset = task_train_dataset
            self.current_task_test_dataset = task_test_dataset
            self.current_task_buffer_dataset = task_train_buffer_dataset
            
            self.seen_train_targets = []
            self.seen_train_classes = []
            for seen_task in range(task):
                self.seen_train_targets.extend([x['global_target'] for x in self.task_sequence[seen_task]])
                self.seen_train_classes.extend([x['classname'] for x in self.task_sequence[seen_task]])
            self.seen_test_targets = copy.deepcopy(self.seen_train_targets)
            self.seen_test_classes = copy.deepcopy(self.seen_train_classes)
            
        return {"train": task_train_dataset, "test": task_test_dataset, "buffer": task_train_buffer_dataset}

    def make_dataloader(self, dset, train=True, custom_batch_size=None):
        custom_collate = None
        if hasattr(dset, "_collate_fn"):
            custom_collate = dset._collate_fn
        
        if train:
            return torch.utils.data.DataLoader(
                dset,
                pin_memory=True,
                drop_last=True,
                batch_size=self.task_batch_size if not custom_batch_size else custom_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=lambda batch: self.collate_fn(batch, custom_collate),
                worker_init_fn=lambda worker_id: np.random.seed(self.args.experiment.seed + self.task * 10000 + worker_id)
            )
        else:
            return torch.utils.data.DataLoader(
                dset,
                pin_memory=True,
                drop_last=False,
                batch_size=self.evaluation_batch_size if not custom_batch_size else custom_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=lambda batch: self.collate_fn(batch, custom_collate),
            )

    def give_task_dataloaders(self, task: int = None):
        set_to_current = task is None

        task_datasets = self.give_task_datasets(task)

        train_loader = self.make_dataloader(task_datasets["train"], train=True)
        test_loader = self.make_dataloader(task_datasets["test"], train=False)
        buffer_loader = None

        if set_to_current:
            task = self.task
                    
        if self.task_buffer_batch_size and task > 0:
            buffer_loader = self.make_dataloader(
                task_datasets["buffer"], train=True, custom_batch_size=self.task_buffer_batch_size)
            
        if set_to_current:
            self.current_train_loader = train_loader
            self.current_test_loader = test_loader
            self.current_buffer_loader = buffer_loader

        return {"train": train_loader, "test": test_loader, "buffer": buffer_loader}

    def give_all_full_test_dataloaders(self):
        """Return full test dataloaders of every unique dataset in self.task_sequence

        Returns:
            dict: {name_of_unique_dataset: test_dataloader}
        """
        return {
            dataset_name: self.make_dataloader(test_dataset, train=False)
            for dataset_name, test_dataset in self.test_datasets.items()
        }

    def give_class_indices_of_current_task_targets(self):
        """Return sorted global targets for current task.

        Returns:
            list: sorted list of all available global targets in a task.
        """
        return sorted([x[-1] for x in self.task_sequence[self.task]])

    def update_dataset_parameters(
        self, data_req_updates: Dict = None, data_params_updates: Dict = None
    ):
        """Takes input arguments to directly manipulate experiment datsets.

        Args:
            data_req_updates (Dict, optional): Directly change dataset attributes. Defaults to None.
            data_params_updates (Dict, optional): Change dataset.PARAMS entries. Defaults to None.
        """
        for dataset_name in self.dataset_names:
            for key, val in data_req_updates.items():
                self.train_datasets[dataset_name].__dict__[key] = val
                self.test_datasets[dataset_name].__dict__[key] = val

            self.train_datasets[dataset_name].PARAMS.update(data_params_updates)
            self.test_datasets[dataset_name].PARAMS.update(data_params_updates)
            self.train_datasets[dataset_name].set_transforms()
            self.test_datasets[dataset_name].set_transforms()

    def summary(self):
        termcolor.cprint("Utilizing multi-dataset aggregation!\n"\
            "Summary Format:\nName [class count]: sample count >> #num_tasks_with_this_dataset.", 
            "yellow", attrs=[])
        
        summary_str = ''
        total_num_classes, total_num_samples = 0, 0

        for dataset_name in self.dataset_names:
            val_counts = [0 if x is None else len(x) for x in self.all_train_idcs[dataset_name]]
            num_samples = np.sum(val_counts)
            num_tasks = np.sum([x > 0 for x in val_counts])
            num_classes = [np.array(self.train_datasets[dataset_name].targets)[x] for x in self.all_train_idcs[dataset_name] if x is not None]
            num_classes = len(np.unique([x for y in num_classes for x in y]))            
            summary_str += f"\n- {dataset_name} [{num_classes}]: {num_samples} >> {num_tasks} tasks."
            total_num_classes += num_classes
            total_num_samples += num_samples
        
        print(summary_str)
        print("----")
        print(f"Total stats: {total_num_classes} classes & {total_num_samples} samples.")
        
    @property
    def checkpoint(self):
        return {"task": self.task, "name": self.name}

    def load_from_checkpoint(self, state_dict):
        self.task = state_dict["task"]
        self.name = state_dict["name"]









