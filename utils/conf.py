import random

import numpy as np
from omegaconf import DictConfig, ListConfig
import torch


def clip_gradients_by_norm(
    continual_learner: torch.nn.Module, grad_clip_norm: float = 0
):
    if grad_clip_norm > 0:
        gradient_norm_clipper = lambda params: (
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            if grad_clip_norm > 0
            else lambda x: x
        )
        # We clip potential non-backbone parameters that are optimized.
        optim_parameters = [x["params"] for x in continual_learner.opt.param_groups]
        optim_parameters = [x for y in optim_parameters for x in y]
        _ = gradient_norm_clipper(optim_parameters)
        # Now we make sure to also clip all relevant base backbone parameters.
        _ = gradient_norm_clipper(continual_learner.backbone.parameters())



                
def freeze_all_but_head(continual_learner: torch.nn.Module):
    assert_str = "Warmup not possible, as no separate head parameters are available!"
    assert hasattr(continual_learner, "head"), assert_str

    # Set requires_grad to False for every optimizable parameter.
    for i in range(len(continual_learner.opt.param_groups)):
        to_opt = continual_learner.opt.param_groups[i]["params"]
        if isinstance(to_opt, list):
            for j in range(len(to_opt)):
                to_opt[j].requires_grad = False
        else:
            to_opt.requires_grad = False

    # Unblock requires_grad for the head parameters.
    for _, weight in continual_learner.head.named_parameters():
        weight.requires_grad = True



def unfreeze_all(continual_learner: torch.nn.Module):
    for i in range(len(continual_learner.to_optimize)):
        to_opt = continual_learner.to_optimize[i]["params"]
        if isinstance(to_opt, list):
            for j in range(len(to_opt)):
                to_opt[j].requires_grad = True
        else:
            to_opt.requires_grad = True



def summarize_args(args, max_len=60):
    key_coll = []
    val_coll = []

    def args_summary(d, basename=""):
        for k, v in d.items():
            if isinstance(v, DictConfig):
                args_summary(v, basename=f"{basename}.{k}")
            else:
                key = f"{basename}.{k}"
                val = v
                key_coll.append(key)
                val_coll.append(val)

    args_summary(args)
    val_coll = [str(x) for x in val_coll]
    max_key_len = np.min([max_len, max([len(x) for x in key_coll])])
    max_val_len = np.min([max_len, max([len(x) for x in val_coll])])
    summary_str = ""
    max_add_len = 0
    for key, val in zip(key_coll, val_coll):
        summary_str += "\n"
        default_key_str = key + " " * (max_key_len - len(key) + 1)
        default_val_str = val + " " * (max_val_len - len(val) + 1)
        if len(val) > max_len:
            default_val_str = default_val_str[: max_len - 6] + " [...] "
        add_str = "| " + default_key_str + "| " + default_val_str + "|"
        if len(add_str) > max_add_len:
            max_add_len = len(add_str)
        summary_str += add_str
    summary_str = "-- Args " + "-" * (max_add_len - 8) + summary_str
    summary_str += "\n" + "-" * max_add_len
    print(summary_str)


   
        
def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return "./data/"


def set_random_seed(seed: int, set_backend: bool = True) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if set_backend:
        torch.backends.cudnn.deterministic = True


##### SANITY CHECK UTILS  
def verify_task_details(args):
    # If dataset.name is not a list (i.e. using a single dataset), convert it to a list regardless.
    if not isinstance(args.experiment.dataset.name, ListConfig):
        args.experiment.dataset.name = [args.experiment.dataset.name]

    if not isinstance(args.experiment.task.num, ListConfig):
        args.experiment.task.num = [args.experiment.task.num]
        if len(args.experiment.task.num) == 1:
            args.experiment.task.num = [
                args.experiment.task.num[0]
                for _ in range(len(args.experiment.dataset.name))
            ]
        if len(args.experiment.task.num) > 1:
            assert_str = f"Please ensure that you provide either a single value for \
                [experiment.task.num] or a list matching the number of datasets used \
                in [experiment.dataset.name]! Currently: experiment.task.num = {args.experiment.task.num}\
                | experiment.dataset.name = {args.experiment.dataset.name}."
            assert len(args.experiment.task.num) == len(
                args.experiment.dataset.name
            ), assert_str    
        torch.backends.cudnn.deterministic = True