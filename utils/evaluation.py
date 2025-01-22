import copy
import json
import os
import pickle
import time
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import termcolor
import torch
import tqdm

import backbones
import continual_lib
import experiment_lib
import utils.loggers
import utils.metrics

plt.switch_backend('agg')

class Evaluator:
    def __init__(
        self,
        args: omegaconf.DictConfig,
        experiment: experiment_lib.PredefinedSequenceExperiment,
        device: torch.device,
        log_folder: str,
        evaluation_only_test_datasets: List[torch.utils.data.Dataset],
        generate_plots: bool=True
    ):
        self.args = args
        self.generate_plots = generate_plots

        os.makedirs(log_folder, exist_ok=True)
        self.log_folder = log_folder

        self.metrics = {
            'task': {metric: utils.metrics.give(metric) for metric in self.args.experiment.evaluation.task_metrics},
            'total': {metric: utils.metrics.give(metric) for metric in self.args.experiment.evaluation.total_metrics}
        }
        self.device = device

        # This will utilize the dataset_names extracted from the provided data sequence.
        self.datasets_to_evaluate = list(args.experiment.dataset.name) + list(args.experiment.evaluation.additional_datasets)

        self.results = self.give_results_dict()
        self.pre_train_results = None

        self.exp_sequence = []

        ### Set up evaluation-only test dataloaders.
        # As in the standard CL experiment, the offset is used to increment target values from other datasets based on
        # how many other classes from other datasets are introduced before them.
        self.evaluation_only_experiments = []
        for evaluation_only_test_dataset in evaluation_only_test_datasets:
            eval_exp = experiment_lib.continual_learning_experiment.EvalOnlyExperiment(
                args, evaluation_only_test_dataset)
            self.evaluation_only_experiments.append(eval_exp)

    def update_dataset_parameters(self, data_req_updates: Dict=None, data_params_updates: Dict=None):
        for i in range(len(self.evaluation_only_experiments)):
            for key, val in data_req_updates.items():
                self.evaluation_only_experiments[i].eval_dataset.__dict__[key] = val
            self.evaluation_only_experiments[i].eval_dataset.PARAMS.update(data_params_updates)
            self.evaluation_only_experiments[i].eval_dataset.set_transforms()

    def give_results_dict(self):
        results = {
            exp: {
                metric_type: {
                    metric_name: [] for metric_name in metric_dict.keys()
                } for metric_type, metric_dict in self.metrics.items()
            } for exp in self.datasets_to_evaluate
        }

        for exp in self.datasets_to_evaluate:
            results[exp]['task_seen'] = []
            results[exp]['exp_task'] = []

        results['task_ids_evaluated_on'] = []
        results['total_experiment_sequence'] = []
        return results

    @property
    def checkpoint(self):
        return {
            'results': self.results,
            'pre_train_results': self.pre_train_results,
        }

    def load_from_checkpoint(self, state_dict):
        self.results = state_dict['results']
        self.pre_train_results = state_dict['pre_train_results']

    def assign_pre_train_results(self, pre_train_results: dict) -> None:
        self.pre_train_results = pre_train_results

    def evaluate_and_summarize(
        self,
        continual_learner: continual_lib.BaseContinualLearner,
        experiment: experiment_lib.BaseExperiment,
        subset_only: float=None
    ):
        result_dict = self.give_results_dict()
        result_dict = self.evaluate(
            continual_learner, experiment, custom_results_dict=result_dict, is_validation=False, subset_only=subset_only)

        train_dset_names = self.args.experiment.dataset.name
        eval_only_dset_names = self.args.experiment.evaluation.additional_datasets
        metrics = self.args.experiment.evaluation.total_metrics

        # Summarize results.
        if len(train_dset_names):
            termcolor.cprint('TRAINING DATASETS', 'yellow', attrs=[])
            print(' | '.join(['Dataset'] + metrics))
            for train_dataset_name in train_dset_names:
                val_coll = [train_dataset_name]
                for metric_name in metrics:
                    metric_dict = result_dict[train_dataset_name]['total'][metric_name]
                    if len(metric_dict):
                        out = metric_dict[0]
                        val_coll.append('{0:3.3f}%'.format(out))
                    else:
                        val_coll.append('--------')
                print(' | '.join(val_coll))

        if len(eval_only_dset_names):
            termcolor.cprint('\nEVAL ONLY DATASETS', 'yellow', attrs=[])
            print(' | '.join(['Dataset'] + metrics))
            for eval_dset_name in eval_only_dset_names:
                val_coll = [eval_dset_name]
                for metric_name in metrics:
                    metric_dict = result_dict[eval_dset_name]['total'][metric_name]
                    if len(metric_dict):
                        out = metric_dict[0]
                        val_coll.append('{0:3.3f}%'.format(out))
                    else:
                        val_coll.append('--------')

                print(' | '.join(val_coll))

        print('\n')
        return result_dict

    def evaluate(
        self,
        continual_learner: continual_lib.BaseContinualLearner,
        experiment: experiment_lib.BaseExperiment,
        custom_results_dict: dict=None,
        is_validation: bool=True,
        subset_only: float=None,
        exclude_from_aggregation: bool=False
    ):

        eval_start_time = time.time()

        if not exclude_from_aggregation:
            results_dict = self.results
        else:
            results_dict = self.give_results_dict()

        if custom_results_dict is not None:
            results_dict = custom_results_dict

        status = continual_learner.training
        continual_learner.eval()

        input_kwargs = {
            'n_tasks': self.args.experiment.task.num
        }

        # Grab each possible full test dataloader for every dataset
        # mentioned in the provided data sequence. Even if only mentioned
        # once with a single class, this will return the full test split.
        all_test_loaders = experiment.give_all_full_test_dataloaders()
        task_capable_loaders = copy.deepcopy(list(all_test_loaders.keys()))
        # Get evaluation-only dataloaders.
        eval_only_loaders = []
        if len(self.evaluation_only_experiments):
            eval_only_loaders = [eval_only_exp.give_test_dataloader() for eval_only_exp in self.evaluation_only_experiments]        
        for i, exp in enumerate(self.args.experiment.evaluation.additional_datasets):
            all_test_loaders[exp] = eval_only_loaders[i]

        subset_only = subset_only if subset_only is not None else self.args.experiment.evaluation.validate_on_subset

        # Update evaluation datasets when evaluating only on subsets of test datasets (subset_only < 1).
        # Save full datasets for later restoration.
        full_datasets = {}
        if subset_only < 1:
            if not exclude_from_aggregation:
                termcolor.cprint('Performing evaluation on {0:3.1f}% subset.'.format(subset_only * 100), 'blue', attrs=[])

            for key in all_test_loaders.keys():
                eval_data = all_test_loaders[key].dataset.data
                eval_preloaded_image_data = all_test_loaders[key].dataset.preloaded_image_data
                eval_targets = all_test_loaders[key].dataset.targets
                full_datasets[key] = [eval_data, eval_preloaded_image_data, eval_targets]

                unique_targets = np.unique(eval_targets)

                # Require at least num_classes * 2 samples over at least 300 samples. Otherwise use full subset.
                subset = np.max([2 * len(unique_targets), 300, int(len(eval_data) * subset_only)])
                subset = np.min([len(eval_data), subset])
                np.random.seed(self.args.experiment.seed)
                num_samples_per_class = int(subset / len(unique_targets))

                # We first sample samples uniformly from each class.
                subset_idcs = []
                for unique_target in unique_targets:
                    target_locs = np.where(np.array(eval_targets) == unique_target)[0]
                    subset_idcs.extend(list(np.random.choice(target_locs, np.min([num_samples_per_class, len(target_locs)]), replace=False)))

                # We fill the remaining spots, if available, randomly.
                samples_left = subset - len(subset_idcs)
                if samples_left > 0:
                    samples_left = list(np.random.choice(list(set(range(len(eval_data))) - set(subset_idcs)), samples_left, replace=False))
                    subset_idcs.extend(samples_left)

                subset_idcs = sorted(subset_idcs)

                if isinstance(eval_data, np.ndarray) or isinstance(eval_data, torch.Tensor):
                    all_test_loaders[key].dataset.data = eval_data[subset_idcs]
                elif isinstance(eval_data, list):
                    all_test_loaders[key].dataset.data = [eval_data[i] for i in subset_idcs]
                if eval_preloaded_image_data:
                    all_test_loaders[key].dataset.preloaded_image_data = [eval_preloaded_image_data[i] for i in subset_idcs]
                if isinstance(eval_targets, np.ndarray) or isinstance(eval_targets, torch.Tensor):
                    all_test_loaders[key].dataset.targets = eval_targets[subset_idcs]
                elif isinstance(eval_targets, list):
                    all_test_loaders[key].dataset.targets = [eval_targets[i] for i in subset_idcs]

        bar_shift = int(exclude_from_aggregation)
        exp_iterator = tqdm.tqdm(all_test_loaders.items(), total=len(all_test_loaders), position=bar_shift, leave=False)
        outputs_coll = {ds: {'predictions': None, 'targets': None} for ds in all_test_loaders.keys()}

        # Precompute all relevant text embeddings to retrieve against later.
        custom_heads = {}
        num_captions = {}
        caption_targets_dict = {}
        if isinstance(continual_learner.head.module, backbones.ClipTextHead):
            for dataset_name, test_loader in tqdm.tqdm(all_test_loaders.items(), desc='Precomputing class/caption embeddings...'):
                is_retrieval_exp = test_loader.dataset.PARAMS['type'] == 'retrieval'
                
                if not is_retrieval_exp:
                    # Simple classification case.
                    classes = test_loader.dataset.PARAMS['classes']
                    with torch.amp.autocast("cuda"), torch.no_grad():
                        embed_coll = continual_learner.head.module.embed_text(classes, self.args.experiment.task.batch_size)
                    custom_heads[dataset_name] = embed_coll
                else:
                    # Retrieval-based dataset evaluation.
                    captions = test_loader.dataset.caption_data
                    caption_targets = []

                    assert_str = f'Captions have to be provided as either list or dict, currently: {type(captions)}!'
                    assert isinstance(captions, list) or isinstance(captions, dict), assert_str

                    # Convert captions that are provided as dict of the form {file_name: caption} into 
                    # a list of [caption_0, ..., caption_N] in order aligned with test_loader.dataset.data.
                    # Note: caption_targets assigns a target to each caption based simply on their idx. 
                    # I.e. the first caption gets 0, the next gets 1 and so on.
                    if isinstance(captions, dict):
                        exp_data = test_loader.dataset.data
                        root_remove_1 = '/'.join(test_loader.dataset.root.split('/')[:-1]) + '/'
                        root_remove_2 = root_remove_1[2:]
                        exp_data = [x.replace(root_remove_1, '').replace(root_remove_2, '') for x in exp_data]

                        if isinstance(exp_data, list) and isinstance(exp_data[0], str):
                            captions = [test_loader.dataset.caption_data[d] for d in exp_data]
                        else:
                            captions = [test_loader.dataset.caption_data[i] for i in range(len(exp_data))]
                        caption_targets = list(range(len(captions)))

                    # If captions are provided in the form of nested lists, expand into single list.
                    # More importantly, ensure that if an image has multiple captions, that the target
                    # in caption_targets is appropriately repeated (e.g. for MSCOCO / FLICKR30k eval).
                    if isinstance(captions[0], list):
                        caption_targets = [[i for _ in range(len(x))] for i, x in enumerate(captions)]
                        caption_targets = [x for y in caption_targets for x in y]
                        captions = [x for y in captions for x in y]
                    else:
                        caption_targets = list(range(len(captions)))

                    with torch.amp.autocast("cuda"), torch.no_grad():
                        embed_coll = continual_learner.head.module.embed_text(captions, self.args.experiment.task.batch_size)

                    custom_heads[dataset_name] = embed_coll
                    num_captions[dataset_name] = len(captions)
                    caption_targets_dict[dataset_name] = caption_targets


        # Compute image embeddings.
        for dataset_name, test_loader in exp_iterator:
            exp_iterator.set_description_str(f'Computing all dataset embeddings [{dataset_name}]')
            batch_iterator = tqdm.tqdm(test_loader, position=1+bar_shift, leave=False, desc='Embedding...')
            targets = np.zeros((len(test_loader.dataset)))
            is_retrieval_exp = test_loader.dataset.PARAMS['type'] == 'retrieval'
            caption_targets = None
            if is_retrieval_exp:
                predictions = np.zeros((len(test_loader.dataset), num_captions[dataset_name]))
                caption_targets = caption_targets_dict[dataset_name]
            else:
                predictions = np.zeros((len(test_loader.dataset), test_loader.dataset.PARAMS['num_classes']))
            count = 0

            for i, data in enumerate(batch_iterator):
                with torch.amp.autocast("cuda"), torch.no_grad():
                    batch_size = len(data['images'])
                    data['images'] = data['images'].to(continual_learner.device)
                    if is_retrieval_exp:
                        assert_str = 'Caption embeddings for retrieval benchmarks need to be precomputed!'
                        assert dataset_name in custom_heads, assert_str

                    features = continual_learner(**data, image_features_only=True)
                    if is_retrieval_exp:
                        features = features / features.norm(dim=-1).reshape(-1, 1)
                    logits = features @ custom_heads[dataset_name].T

                    predictions[count:count + batch_size] = logits.data.detach().cpu().numpy()
                    targets[count:count + batch_size] = data['targets'].detach().cpu().numpy()

                count += batch_size

            outputs_coll[dataset_name]['predictions'] = predictions
            # We normalize target values ([0, ..., C-1]) for evaluation as we operate on a per-dataset basis.
            if is_retrieval_exp:
                outputs_coll[dataset_name]['targets'] = caption_targets - np.min(targets)
            else:
                outputs_coll[dataset_name]['targets'] = targets - np.min(targets)

        if not exclude_from_aggregation:
            print('\n')

        ### Compute Stage 0 Metrics (i.e. metrics that don't rely on other metrics).
        # Compute Metrics over full test dataset outputs.
        metric_dict = self.metrics['total']

        for metric_name, metric in metric_dict.items():
            if metric.stage == 0:
                for dset in all_test_loaders.keys():
                    dataset_type = all_test_loaders[dset].dataset.PARAMS['type']
                    input_kwargs['predictions'] = outputs_coll[dset]['predictions']
                    input_kwargs['targets'] = outputs_coll[dset]['targets']
                    input_kwargs['device'] = continual_learner.device

                    if metric._can_be_applied(dataset_type):
                        metric_val = metric(**input_kwargs)
                        results_dict[dset]['total'][metric_name].append(metric_val[metric_name])

        # Compute Metrics over task-specific test dataset outputs. 
        # Currently, we only compute task-specific metrics when a single dataset is utilized throughout each task.
        compute_task_metrics = self.args.experiment.task.num > 1
        compute_task_metrics &= self.args.experiment.evaluation.validate_on_subset == 1
        compute_task_metrics &= len(self.args.experiment.dataset.name) == 1

        if compute_task_metrics:
            # For each test dataset, we get the respective sample indices that correspond to each task.
            test_task_indices = {
                dataset_name: [experiment.get_task_indices(dataset_name, task) for task in range(self.args.experiment.task.num)]
                for dataset_name in task_capable_loaders
            }
            
            metric_dict = self.metrics['task']
            for metric_name, metric in metric_dict.items():
                if metric.stage == 0:
                    for dataset_name in task_capable_loaders:
                        dataset_type = experiment.test_datasets[dataset_name].PARAMS['type']
                        metric_val_coll = []
                        if metric._can_be_applied(dataset_type):
                            for task, task_idcs in enumerate(test_task_indices[dataset_name]):
                                input_kwargs['predictions'] = outputs_coll[dataset_name]['predictions'][task_idcs]
                                input_kwargs['targets'] = outputs_coll[dataset_name]['targets'][task_idcs]
                                input_kwargs['task_index'] = task
                                metric_val = metric(**input_kwargs)
                                metric_val_coll.append(metric_val[metric_name])
                            results_dict[dataset_name]['task'][metric_name].append(metric_val_coll)

        if custom_results_dict is None and not exclude_from_aggregation:
            self.summarize_current_results(experiment, all_test_loaders)

        continual_learner.train(status)

        # Restore full datasets if subset_only < 1.
        if subset_only < 1 and full_datasets:
            for key in all_test_loaders.keys():
                all_test_loaders[key].dataset.data = full_datasets[key][0]
                all_test_loaders[key].dataset.preloaded_image_data = full_datasets[key][1]
                all_test_loaders[key].dataset.targets = full_datasets[key][2]

        if not exclude_from_aggregation:
            print('Finished evaluation in {0:4.2f}s.\n'.format(time.time() - eval_start_time))

        return results_dict

    def summarize_current_results(
        self,
        experiment: experiment_lib.PredefinedSequenceExperiment,
        all_test_loaders: dict
    ):
        if self.args.experiment.task.num > 1:
            termcolor.cprint(f'\nEvaluation Current Task {experiment.task + 1}/{self.args.experiment.task.num}:', 'white', attrs=["underline"])
            curr_task = experiment.task
            task_dataset_flags = [experiment.all_train_idcs[dataset_name][curr_task] is not None for dataset_name in experiment.dataset_names]
            task_datasets = np.array(experiment.dataset_names)[task_dataset_flags]
            num_datasets = sum(task_dataset_flags)
            termcolor.cprint(f'Covering {num_datasets} adaptation dataset(s).\n', 'yellow', attrs=[])


            ### Summarize performance on full test data.            
            for task_dataset in task_datasets:
                task_dataset_total_results = self.results[task_dataset]['total']
                valid_results = {}
                for k, v in task_dataset_total_results.items():
                    if v and k in self.metrics['total']:
                        valid_results[k] = v
                for metric_name, metric_value in valid_results.items():
                    res_arr = np.array(metric_value)[curr_task]
                    if not isinstance(res_arr, list):
                        res_arr = [res_arr]
                    res = ['{0:3.2f}%'.format(x) for x in res_arr]
                    termcolor.cprint(f'{task_dataset.capitalize()} - Total {metric_name.capitalize()}:', 'blue', attrs=[])
                    if self.pre_train_results is not None:
                        pre_train_metric_value = self.pre_train_results[task_dataset]['total'][metric_name]
                        if pre_train_metric_value:
                            res = ['[ZS: {0:3.2f}%]'.format(pre_train_metric_value[-1])] + res
                    print(" -> ".join(res))     



            ### Summarize performance on task-chunked test data.
            eval_task_metrics = self.args.experiment.evaluation.validate_on_subset == 1
            eval_task_metrics &= len(self.args.experiment.dataset.name) == 1

            if eval_task_metrics:
                task_dataset_task_results = self.results[task_dataset]["task"]
                valid_results = {}
                for k, v in task_dataset_task_results.items():
                    if v and k in self.metrics['task']:
                        valid_results[k] = v
                for metric_name, metric_value in valid_results.items():
                    res = ['{0:3.2f}%'.format(x) for x in metric_value[-1]]
                    termcolor.cprint(f'Current Task {metric_name.capitalize()}:', 'blue', attrs=[])
                    print(" | ".join(res))
                    res_arr = np.array(metric_value)
                    progr = []
                    for i in range(len(res_arr)):
                        avg_task_perf_to_i = np.mean(res_arr[i, :i+1])
                        progr.append('{0:3.2f}%'.format(avg_task_perf_to_i))
                    termcolor.cprint(f'Avg. Task {metric_name.capitalize()} Progression:', 'blue', attrs=[])
                    print(' -> '.join(progr))
            print('\n')


                                   
        termcolor.cprint('\nEvaluation On All Datasets:', 'white', attrs=["underline"])
        self.avg_results = {metric_name: {} for metric_name in self.metrics['total'].keys()}
                
        for metric_name, metric_model in self.metrics['total'].items():
            avg_metric, eval_only_metric, train_metric = [], [], []
            pre_avg_metric, pre_eval_only_metric, pre_train_metric = [], [], []
            
            for dataset_name in self.datasets_to_evaluate:
                dataset_type = all_test_loaders[dataset_name].dataset.PARAMS['type']                
                if metric_model._can_be_applied(dataset_type):
                    metric_val = self.results[dataset_name]['total'][metric_name]

                    avg_metric.append(metric_val)
                    if dataset_name in self.args.experiment.evaluation.additional_datasets:
                        eval_only_metric.append(metric_val)
                    else:
                        train_metric.append(metric_val)
            if avg_metric:
                avg_metric = list(np.array(avg_metric).mean(axis=0))
                eval_only_metric = [] if not eval_only_metric else list(np.array(eval_only_metric).mean(axis=0))
                train_metric = [] if not train_metric else list(np.array(train_metric).mean(axis=0))

            pre_avg_metric, pre_eval_only_metric, pre_train_metric = [], [], []
            if self.pre_train_results is not None and avg_metric:
                for dataset_name in self.datasets_to_evaluate:
                    dataset_type = all_test_loaders[dataset_name].dataset.PARAMS['type']                
                    if metric_model._can_be_applied(dataset_type):            
                        pre_metric = self.pre_train_results[dataset_name]['total'][metric_name]
                        if pre_metric:
                            pre_avg_metric.append(pre_metric[-1])
                            if dataset_name in self.args.experiment.evaluation.additional_datasets:
                                pre_eval_only_metric.append(pre_metric[-1])
                            else:
                                pre_train_metric.append(pre_metric[-1])

                pre_avg_metric = [] if not pre_avg_metric else np.mean(pre_avg_metric)
                pre_eval_only_metric = [] if not pre_eval_only_metric else np.mean(pre_eval_only_metric)
                pre_train_metric = [] if not pre_train_metric else np.mean(pre_train_metric)

            self.avg_results[metric_name]['full_adapt'] = avg_metric
            self.avg_results[metric_name]['eval_only_adapt'] = eval_only_metric
            self.avg_results[metric_name]['train_adapt'] = train_metric
            self.avg_results[metric_name]['full_zs'] = pre_avg_metric
            self.avg_results[metric_name]['eval_only_zs'] = pre_eval_only_metric
            self.avg_results[metric_name]['train_zs'] = pre_train_metric

        ### Showcase progression over total average!
        for metric_name, metric_results in self.avg_results.items():
            if len(self.avg_results[metric_name]['full_adapt']):
                print_train_metric, print_eval_only_metric = '[N/A]', '[N/A]'
                if len(metric_results['full_adapt']) <= 3:
                    print_avg_metric = ' -> '.join(['{0:3.2f}%'.format(x) for x in metric_results['full_adapt']])
                    if metric_results['train_adapt']:
                        print_train_metric = ' -> '.join(['{0:3.2f}%'.format(x) for x in metric_results['train_adapt']])
                    if metric_results['eval_only_adapt']:
                        print_eval_only_metric = ' -> '.join(['{0:3.2f}%'.format(x) for x in metric_results['eval_only_adapt']])
                else:
                    print_avg_metric = '{0:3.2f}% [Best: {1:3.2f}%]'.format(metric_results['full_adapt'][-1], np.max(metric_results['full_adapt']))
                    if metric_results['train_adapt']:
                        print_train_metric = '{0:3.2f}% [Best: {1:3.2f}%]'.format(metric_results['train_adapt'][-1], np.max(metric_results['train_adapt']))
                    if metric_results['eval_only_adapt']:
                        print_eval_only_metric = '{0:3.2f}% [Best: {1:3.2f}%]'.format(metric_results['eval_only_adapt'][-1], np.max(metric_results['eval_only_adapt']))

                termcolor.cprint(f'Average {metric_name.capitalize()} across [all | train | eval_only] datasets:', 'blue', attrs=[])
                full_pre_zs = '[No ZS]' if not metric_results['full_zs'] else '[ZS: {0:3.2f}] -> '.format(metric_results['full_zs'])
                train_pre_zs = '[No ZS]' if not metric_results['train_zs'] else '[ZS: {0:3.2f}] -> '.format(metric_results['train_zs'])
                eval_only_pre_zs = '[No ZS]' if not metric_results['eval_only_zs'] else '[ZS: {0:3.2f}] -> '.format(metric_results['eval_only_zs'])

                print('{0}{1} | {2}{3} | {4}{5}'.format(
                    full_pre_zs, print_avg_metric, train_pre_zs, print_train_metric, eval_only_pre_zs, print_eval_only_metric))
            else:
                termcolor.cprint(f'No results for {metric_name.capitalize()} across all datasets', 'blue', attrs=[])
            print('\n')


    def write_results(self, format: str='json', custom_data: Any=None, custom_name: str=None):
        data_to_store = self.results if custom_data is None else custom_data
        store_name = 'evaluation_results' if custom_name is None else custom_name
        store_name += f'.{format}'

        if format == 'json':
            base_file = os.path.join(self.log_folder, store_name)
            json.dump(data_to_store, open(base_file, 'w'), indent=6)

            if self.pre_train_results is not None and custom_data is None:
                zeroshot_file = os.path.join(self.log_folder, 'evaluation_zeroshot_results.json')
                json.dump(self.pre_train_results, open(zeroshot_file, 'w'), indent=6)

        if format == 'pkl':
            base_file = os.path.join(self.log_folder, 'evaluation_results.pkl')
            pickle.dump(data_to_store, open(base_file, 'wb'))
            if self.pre_train_results is not None and custom_data is None:
                zeroshot_file = os.path.join(self.log_folder, 'evaluation_zeroshot_results.pkl')
                pickle.dump(self.pre_train_results, open(zeroshot_file, 'wb'))

    def prepare_log_dict(self, custom_log_data: dict=None, use_pre_train_results: bool=False, merge_to_train_results: bool=True, custom_preemble: str=None):
        wandb_log_dict = {}

        if custom_log_data is None:
            results_to_log = self.pre_train_results if use_pre_train_results else self.results
        else:
            results_to_log = custom_log_data

        preemble = 'zeroshot' if use_pre_train_results else 'adaptation'

        if merge_to_train_results:
            if use_pre_train_results:
                print('\n[W&B logging] Zeroshot results are stored alongside adaptation results (first value).')
            preemble = 'adaptation'

        if custom_preemble is not None:
            preemble = custom_preemble
        
        for exp, exp_dict in results_to_log.items():
            if exp not in ['task_ids_evaluated_on', 'total_experiment_sequence']:
                for mode, mode_dict in exp_dict.items():
                    if mode not in ['task_seen', 'exp_task']:
                        for metric in mode_dict.keys():
                            vals = results_to_log[exp][mode][metric]
                            if len(vals):
                                if mode == 'task':
                                    for i, subval in enumerate(vals[-1]):
                                        wandb_log_dict[f'{preemble}.{exp}.{mode}-{i+1}.{metric}'] = subval
                                    wandb_log_dict[f'{preemble}.{exp}.{mode}.mean_{metric}'] = np.mean(vals[-1])
                                elif mode == 'total':
                                    wandb_log_dict[f'{preemble}.{exp}.{mode}.{metric}'] = vals[-1]
                                    
        for metric_name in self.metrics['total'].keys():
            exp_val = {}
            full_vals, eval_only_vals, train_vals = [], [], []
            for exp, exp_dict in results_to_log.items():
                if exp not in ['task_ids_evaluated_on', 'total_experiment_sequence']:
                    if exp_dict['total'][metric_name]:
                        exp_val[exp] = exp_dict['total'][metric_name][-1]
            if exp_val:
                for ds in exp_val.keys():
                    full_vals.append(exp_val[ds])
                    if ds in self.args.experiment.evaluation.additional_datasets:
                        eval_only_vals.append(exp_val[ds])
                    else:
                        train_vals.append(exp_val[ds])

            full_ret_val = np.nan if not full_vals else np.mean(full_vals)
            wandb_log_dict[f'{preemble}.alldata-full.total.{metric_name}'] = full_ret_val
            train_ret_val = np.nan if not train_vals else np.mean(train_vals)
            wandb_log_dict[f'{preemble}.alldata-trainonly.total.{metric_name}'] = train_ret_val
            eval_ret_val = np.nan if not eval_only_vals else np.mean(eval_only_vals)
            wandb_log_dict[f'{preemble}.alldata-evalonly.total.{metric_name}'] = eval_ret_val
            traineval_geomean = np.sqrt(train_ret_val * eval_ret_val)
            wandb_log_dict[f'{preemble}.alldata-traineval_geomean.total.{metric_name}'] = traineval_geomean
        
        return wandb_log_dict
