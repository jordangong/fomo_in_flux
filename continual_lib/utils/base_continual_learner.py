import contextlib
import copy
from typing import List, Any

import numpy as np
import omegaconf
import torch.nn as nn
import torch.optim
import torch

import backbones
import continual_lib
import continual_lib.utils
import experiment_lib


class BaseContinualLearner(nn.Module):
    """
    Continual learning model.
    """

    # False if method only aggregates data for each task with no learning happening during task streaming.
    ON_TASK_LEARNING = True
    # Only true if instead of training on task sequences, training on full dataset aggregation is done.
    JOINT_LEARNING = False
    # If (e.g. for SSL) auxiliarly augmented input images are needed.
    REQ_AUX_INPUTS = False
    # If non-augmented input images are needed (e.g. for buffer methods that perform their own augmentation.)
    REQ_NON_AUG_INPUTS = True
    # Denotes whether a method supports both open-set and closed-set continual learning
    SUPPORTED_TRAINING_MODES = [
        "classification_default",
        "classification_dataset",
        "classification_task",
        "classification_seen",
        "contrastive",
    ]
    # Available optimization methods
    AVAILABLE_OPTIMIZER = ["sgd", "adam", "adamw", "galore_adamw"]
    # Available learning rate scheduling methods
    AVAILABLE_SCHEDULER = [
        "none", 
        "multistep",
        "cosine",
        "cosine_with_linear_warmup",
        "rsqrt",
        "rsqrt_with_recovery",
        "cosine_with_linear_warmup_and_recovery",
        "cosine_with_linear_warmup_and_dynamic_recovery",
        "wsd"
    ]

    def __init__(
        self,
        args: omegaconf.DictConfig,
        backbone: nn.Module,
        head: nn.Module,
        loss: Any,
        device: torch.device,
    ) -> None:
        super(BaseContinualLearner, self).__init__()

        ### Default attributes and flags.
        self.args = args
        self.backbone = backbone
        self.backbone_feature_dim = backbones.feature_dim_dict[
            self.args.experiment.backbone.name
        ]
        self.head = head
        self.loss = loss
        self.device = device
        self.training_mode = self.args.experiment.training
        self.freeze_features = self.args.experiment.backbone.freeze_features
        self.freeze_head = self.args.experiment.backbone.freeze_head
        self.transform = None

        ### Connect various desired hooks to backbone and head modules.
        # By default (based on backbones.default_heads), at least a "features"-hook will be connected.
        # This hook contains what is generally regarded as the "feature"-output of the model
        # (e.g. the input to the final linear layer in standard classification networks).
        self.backbone_hook_handles, self.backbone_hook_dict = [], {}
        self.backbone_hooks_required = [
            x for x in args.continual.hook_to if x.split(".")[0] == "backbone"
        ]
        self.head_hook_handles, self.head_hook_dict = [], {}
        self.head_hooks_required = [
            x for x in args.continual.hook_to if x.split(".")[0] == "head"
        ]
        self._hook_backbone_and_head()
        self.backbone_hook_handles.append(
            continual_lib.utils.hook_default_features(
                args, self.backbone.module, self.backbone_hook_dict
            )
        )

        ### Continual Learning specific parameters.
        self.global_tasks_seen = []
        self.seen_targets = []
        self.global_mask_idcs = []

        ### Define parameters to optimize for:
        self._set_optim_params()

        ### Move itself to device.
        self.to(self.device)

        ### Initialize task counter.
        self.task = None

        ### (Optional) Weight Storage for Merging.
        # Initial training weights.
        self.checkpoint_storage = {
            "init": {
                "backbone": copy.deepcopy({k: v.cpu() for k, v in self.backbone.state_dict().items()}),
                "head": copy.deepcopy({k: v.cpu() for k, v in self.head.state_dict().items()}),
            },
            "train": {}, "running": {}, "eval": {},
        }
        # Running task training weights (list).
        self.checkpoint_storage["running"] = {
            "backbone": [copy.deepcopy(self.checkpoint_storage["init"]["backbone"])],
            "head": [copy.deepcopy(self.checkpoint_storage["init"]["head"])],
        }
        # Training by default begins with the last list entry.
        self.checkpoint_storage["train"]["backbone"] = copy.deepcopy(self.checkpoint_storage["running"]["backbone"][-1])
        self.checkpoint_storage["train"]["head"] = copy.deepcopy(self.checkpoint_storage["running"]["head"][-1])

        # Weights to use for evaluation, e.g. latest:
        self.checkpoint_storage["eval"]["backbone"] = copy.deepcopy(self.checkpoint_storage["running"]["backbone"][-1])
        self.checkpoint_storage["eval"]["head"] = copy.deepcopy(self.checkpoint_storage["running"]["head"][-1])


    def _set_optim_params(self):
        self.to_optimize = []
        if not self.freeze_features:
            self.to_optimize.append({"params": self.backbone.parameters()})
        if not self.freeze_head:
            self.to_optimize.append({"params": self.head.parameters()})
        
    def _hook_backbone_and_head(self):
        handles = self.backbone_hook_handles + self.head_hook_handles
        for handle in handles:
            handle.remove()

        hooks_needed = self.backbone_hooks_required + self.head_hooks_required
        for hook in hooks_needed:
            hook_dict = (
                self.backbone_hook_dict if "backbone." in hook else self.head_hook_dict
            )
            hook_location = self._find_hook_location(hook)
            forward_hook = continual_lib.utils.ForwardHook(hook_dict, hook)
            if isinstance(hook_location, torch.nn.Sequential):
                self.backbone_hook_handles.append(
                    hook_location[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone_hook_handles.append(
                    hook_location.register_forward_hook(forward_hook)
                )

    def _find_hook_location(self, hook):
        # Hook locations always start with backbone.[...] or head.[...].
        # We exclude these, as they are given by default.
        hook_path = hook.split(".")[1:]
        if "backbone." in hook:
            hook_location = self.backbone.module
        elif "head." in hook:
            hook_location = self.head.module
        for subhook in hook_path:
            if "[" in subhook:
                # If we have to index the module.
                base_subhook, index = subhook.split("[")
                index = int(index.split("]")[0])
                hook_location = hook_location.__dict__["_modules"][base_subhook][index]
            else:
                hook_location = hook_location.__dict__["_modules"][subhook]
        return hook_location

    @staticmethod
    def put_gpu(state_dict):
        return {key: val.cuda() for key, val in state_dict.items()}

    def define_training_weights(self, **kwargs):
        for mode in ["backbone", "head"]:
            self.checkpoint_storage["train"][mode] = copy.deepcopy(self.checkpoint_storage["running"][mode][-1])

    def define_evaluation_weights(self, **kwargs):
        for mode in ["backbone", "head"]:
            self.checkpoint_storage["eval"][mode] = copy.deepcopy(self.checkpoint_storage["running"][mode][-1])

    def prepare_for_training(
        self,
        experiment: experiment_lib.PredefinedSequenceExperiment = None,
        **kwargs
    ):
        """Initialize model weights for training on a given task.
        
        By default, we simple continue with the model weights trained from previous tasks.
        """
        self.define_training_weights()
        self.backbone.load_state_dict(self.put_gpu(self.checkpoint_storage["train"]["backbone"]))
        self.head.load_state_dict(self.put_gpu(self.checkpoint_storage["train"]["head"]))

    def prepare_for_evaluation(
        self,
        experiment: experiment_lib.PredefinedSequenceExperiment = None,
        eval_params: dict = None,
        **kwargs
    ):
        """Initialize model weights for evaluation.
        
        By default, we simple continue with the model weights trained from previous tasks.
        """
        self.define_evaluation_weights()
        self.backbone.load_state_dict(self.put_gpu(self.checkpoint_storage["eval"]["backbone"]))
        self.head.load_state_dict(self.put_gpu(self.checkpoint_storage["eval"]["head"]))

    def begin_task(
        self,
        optimizer: str = None,
        scheduler: str = None,
        scheduler_steps: int = None,
        milestone_steps: List[int] = None
    ) -> None:
        
        if self.task is None:
            self.task = 0
        else:
            self.task += 1
            
        self.initialize_optimizer_and_scheduler(
            optimizer, scheduler, scheduler_steps, milestone_steps
        )

    def initialize_optimizer_and_scheduler(
        self,
        optimizer: str = None,
        scheduler: str = None,
        scheduler_steps: int = None,
        milestone_steps: List[int] = None,
    ) -> None:
        if optimizer is None:
            optimizer = self.args.experiment.optimizer.name
        if scheduler is None:
            scheduler = self.args.experiment.scheduler.name
        if scheduler_steps is None:
            scheduler_steps = self.args.experiment.task.n_epochs

        # Set up optimizer.
        assert_str = f"No optimizer {optimizer} available. Please choose from {self.AVAILABLE_OPTIMIZER}."
        assert optimizer in self.AVAILABLE_OPTIMIZER, assert_str

        if not hasattr(self, "to_optimize_summary"):
            self.to_optimize_summary = [
                {"default_lr": False, "default_decay": False}
                for _ in range(len(self.to_optimize))
            ]

        for i in range(len(self.to_optimize)):
            if (
                "lr" not in self.to_optimize[i]
                or self.to_optimize_summary[i]["default_lr"]
            ):
                # If no explicit learning rate is given for a parameter-set, we utilize the default learning rate.
                self.to_optimize[i]["lr"] = self.args.experiment.optimizer.lr
                self.to_optimize_summary[i]["default_lr"] = True
            if (
                "weight_decay" not in self.to_optimize[i]
                or self.to_optimize_summary[i]["default_decay"]
            ):
                # If no explicit weight decay is given for a parameter-set, we utilize the default weight decay.
                self.to_optimize[i][
                    "weight_decay"
                ] = self.args.experiment.optimizer.weight_decay
                self.to_optimize_summary[i]["default_decay"] = True

        if optimizer == "sgd":
            self.opt = torch.optim.SGD(self.to_optimize)
        elif optimizer == "adamw":
            self.opt = torch.optim.AdamW(self.to_optimize)
        elif optimizer == "galore_adamw":
            try:
                from galore_torch import GaLoreAdamW
            except:
                raise ModuleNotFoundError('Please install galore_torch via <pip install galore-torch>!')
            param_dicts = []
            if not isinstance(self.to_optimize[0], list):
                self.to_optimize = [{key: x if key != 'params' else list(x) for key, x in param_dict.items()} for param_dict in self.to_optimize]

            for i in range(len(self.to_optimize)):
                for params in self.to_optimize[i]['params']:
                    if params.ndim != 2:
                        param_config = {'params': params}
                    else:
                        param_config = {
                            'params': params,
                            'rank': self.args.continual.galore.rank,
                            'update_proj_gap': self.args.continual.galore.update_proj_gap,
                            'scale': self.args.continual.galore.scale,
                            'proj_type': self.args.continual.galore.proj_type                                    
                        }                        
                    param_dicts.append(param_config)
            self.opt = GaLoreAdamW(param_dicts, lr = self.to_optimize[0]['lr'], weight_decay=self.to_optimize[0]['weight_decay'], no_deprecation_warning=True)
        elif optimizer == "adam":
            self.opt = torch.optim.AdamW(self.to_optimize)

        ### Mixed Precision Scaler.
        self.scaler = torch.amp.GradScaler('cuda')

        # Set up scheduler.
        assert_str = f"No scheduler {scheduler} available. Please choose from {self.AVAILABLE_SCHEDULER}."
        assert scheduler in self.AVAILABLE_SCHEDULER, assert_str

        # Cosine base scheduler with warmup.
        def cosine_with_linear_warmup(step, custom_steps=None, norm=True):
            if custom_steps:
                scheduler_steps = custom_steps
            norm_steps = step
            if norm:
                norm_steps = step % scheduler_steps
            warm_steps = int(
                np.ceil(
                    self.args.experiment.scheduler.warmup_perc * scheduler_steps
                )
            )
            if warm_steps and norm_steps < warm_steps:
                warm_epoch = norm_steps % warm_steps
                lrval = (
                    self.args.experiment.optimizer.lr
                    * (warm_epoch + 1)
                    / warm_steps
                )
            else:
                e = (norm_steps - warm_steps) + 1
                es = scheduler_steps - warm_steps
                mul = 0.5 * (1 + np.cos(np.pi * e / es))
                min_lr = (
                    self.args.experiment.optimizer.lr
                    * self.args.experiment.scheduler.cosine_lr_mul
                )
                lrval = min_lr + mul * (self.args.experiment.optimizer.lr - min_lr)
            return lrval * 1 / self.args.experiment.optimizer.lr

        # Rsqrt base scheduler with warmup.
        def rsqrt(step):
            norm_steps = step % scheduler_steps
            warmup_init_lr = self.args.experiment.optimizer.lr * self.args.experiment.scheduler.cosine_lr_mul
            warmup_steps = self.args.experiment.scheduler.warmup_perc
            if self.args.experiment.scheduler.warmup_perc < 1:
                warmup_steps = int(np.ceil(self.args.experiment.scheduler.warmup_perc * scheduler_steps))
            cooldown_steps = self.args.experiment.scheduler.cooldown_perc
            if self.args.experiment.scheduler.cooldown_perc < 1:
                cooldown_steps = int(np.ceil(self.args.experiment.scheduler.cooldown_perc * scheduler_steps))
            cooldown_start = scheduler_steps - cooldown_steps

            if warmup_steps and norm_steps < warmup_steps:
                lrval = self.args.experiment.optimizer.lr * (norm_steps + 1) / warmup_steps
            elif cooldown_steps and norm_steps >= cooldown_start:
                cool_epoch = (norm_steps - cooldown_start) % cooldown_steps
                mul = warmup_steps ** 0.5 / (norm_steps) ** 0.5
                ref_lr = warmup_init_lr + mul * (self.args.experiment.optimizer.lr - warmup_init_lr)        
                lrval = (ref_lr * (cooldown_steps - cool_epoch - 1) / cooldown_steps)
            else:
                mul = warmup_steps ** 0.5 / (norm_steps) ** 0.5
                lrval = warmup_init_lr + mul * (self.args.experiment.optimizer.lr - warmup_init_lr)
            return lrval / self.args.experiment.optimizer.lr

        def rsqrt_recovered(step):
            norm_steps = step % scheduler_steps

            warmup_steps = self.args.experiment.scheduler.warmup_perc
            if self.args.experiment.scheduler.warmup_perc < 1:
                warmup_steps = int(np.ceil(self.args.experiment.scheduler.warmup_perc * scheduler_steps))
            # seen_warmup_steps += warmup_steps
            cooldown_steps = self.args.experiment.scheduler.cooldown_perc
            if self.args.experiment.scheduler.cooldown_perc < 1:
                cooldown_steps = int(np.ceil(self.args.experiment.scheduler.cooldown_perc * scheduler_steps))
            
            # seen_cooldown_steps + cooldown_steps
            cooldown_start = scheduler_steps - cooldown_steps

            warmup_init_lr = self.args.experiment.optimizer.lr * self.args.experiment.scheduler.cosine_lr_mul
            
            vmul = 1
            if self.task:
                if self.args.experiment.scheduler.recovery_mode == 'continued':
                    seen_steps = (self.task) * (scheduler_steps - cooldown_steps) 
                    seen_steps -= (self.task - 1) * warmup_steps
                    vmul = warmup_steps ** 0.5 / (seen_steps) ** 0.5    
                elif self.args.experiment.scheduler.recovery_mode == 'autoregressive':
                    seen_steps = (self.task) * (scheduler_steps) + warmup_steps
                    vmul = ((self.task + 1)* warmup_steps) ** 0.5 / (seen_steps) ** 0.5                
            base_lr = warmup_init_lr + vmul * (self.args.experiment.optimizer.lr - warmup_init_lr)
            
            if self.args.experiment.scheduler.recovery_mode != 'recovered':
                f_mul = lambda warmup_steps, norm_steps: warmup_steps ** 0.5 / norm_steps ** 0.5
            else:
                f_mul = lambda warmup_steps, norm_steps: warmup_steps ** 0.5 / (norm_steps + self.task * (scheduler_steps - cooldown_steps - warmup_steps)) ** 0.5
                
            if warmup_steps and norm_steps < warmup_steps:
                if self.args.experiment.scheduler.recovery_mode == 'recovered' and self.task:
                    vmul = f_mul(warmup_steps, self.task * (scheduler_steps - cooldown_steps - warmup_steps))
                    base_lr = warmup_init_lr + vmul * (self.args.experiment.optimizer.lr - warmup_init_lr)  
                lrval = (base_lr * (norm_steps + 1) / warmup_steps)
            elif cooldown_steps and norm_steps >= cooldown_start:
                cool_epoch = (norm_steps - cooldown_start) % cooldown_steps
                vmul = f_mul(warmup_steps, norm_steps)
                ref_lr = warmup_init_lr + vmul * (base_lr - warmup_init_lr)        
                lrval = (ref_lr * (cooldown_steps - cool_epoch -1) / cooldown_steps)       
            else:
                vmul = f_mul(warmup_steps, norm_steps)
                lrval = warmup_init_lr + vmul * (base_lr - warmup_init_lr)
            return lrval / self.args.experiment.optimizer.lr

        def cosine_with_linear_warmup_recovered(step):
            norm_steps = step % scheduler_steps
            warm_steps = int(
                np.ceil(
                    self.args.experiment.scheduler.warmup_perc * scheduler_steps
                )
            )
            min_lr = self.args.experiment.optimizer.lr * self.args.experiment.scheduler.cosine_lr_mul
            
            vmul = 1

            if self.task:
                if self.args.experiment.scheduler.recovery_mode == 'autoregressive':
                    e = self.task * (scheduler_steps - warm_steps) + 1
                    es = (self.task + 1) * (scheduler_steps - warm_steps)
                    vmul = 0.5 * (1 + np.cos(np.pi * e / es))          
            base_lr = min_lr + vmul * (self.args.experiment.optimizer.lr - min_lr)
            
            if warm_steps and norm_steps < warm_steps:
                warm_epoch = norm_steps % warm_steps
                lrval = base_lr * (warm_epoch + 1) / warm_steps
            else:
                e = (norm_steps - warm_steps) + 1
                es = scheduler_steps - warm_steps
                vmul = 0.5 * (1 + np.cos(np.pi * e / es))
                lrval = min_lr + vmul * (base_lr - min_lr)
                                
            return lrval / self.args.experiment.optimizer.lr

        def cosine_with_linear_warmup_recovered_dynamic(step):
            norm_steps = step % scheduler_steps
            warm_steps = int(
                np.ceil(
                    self.args.experiment.scheduler.warmup_perc * scheduler_steps
                )
            )
            min_lr = self.args.experiment.optimizer.lr * self.args.experiment.scheduler.cosine_lr_mul
            
            vmul = 1

            if self.task:
                if self.args.experiment.scheduler.recovery_mode == 'autoregressive':
                    e = self.task * (scheduler_steps - warm_steps) + 1
                    es = (self.task + 1) * (scheduler_steps - warm_steps)
                    vmul = 0.5 * (1 + np.cos(np.pi * e / es))
            base_lr = min_lr + vmul * (self.args.experiment.optimizer.lr - min_lr)
            
            if warm_steps and norm_steps < warm_steps:
                warm_epoch = norm_steps % warm_steps
                lrval = base_lr * (warm_epoch + 1) / warm_steps
            else:
                lrval = cosine_with_linear_warmup(self.task * scheduler_steps + norm_steps, scheduler_steps * (self.task+1), norm=False) * self.args.experiment.optimizer.lr

            return lrval / self.args.experiment.optimizer.lr

        # Intialize pytorch scheduler.                
        if scheduler == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.opt,
                milestones=milestone_steps,
                gamma=self.args.experiment.scheduler.multistep_scale,
            )

        elif scheduler == "none":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.opt, milestones=[100000000000], gamma=1
            )

        elif scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                T_max=scheduler_steps,
                eta_min=self.args.experiment.optimizer.lr
                * self.args.experiment.scheduler.cosine_lr_mul,
            )

        elif scheduler == "cosine_with_linear_warmup":
            assert_str = "Number of warmup epochs bigger than total epoch number!"
            assert self.args.experiment.scheduler.warmup_perc < 1, assert_str

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=lambda step: cosine_with_linear_warmup(step, scheduler_steps)
            )

        elif scheduler == "rsqrt":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=lambda step: rsqrt(step)
            )
                

        elif scheduler == "rsqrt_with_recovery":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=lambda step: rsqrt_recovered(step)
            )

        elif scheduler == "cosine_with_linear_warmup_and_recovery":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=lambda step: cosine_with_linear_warmup_recovered(step)
            )        
        elif scheduler == "cosine_with_linear_warmup_and_dynamic_recovery":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=lambda step: cosine_with_linear_warmup_recovered_dynamic(step)
            )

    def end_task(
        self,
        experiment: experiment_lib.PredefinedSequenceExperiment = None,
        **kwargs,
    ):
        with torch.no_grad():
            # Store latest training weights.
            self.checkpoint_storage["running"]["backbone"].append(copy.deepcopy(self.backbone.state_dict()))
            self.checkpoint_storage["running"]["head"].append(copy.deepcopy(self.head.state_dict()))

    def gradient_update(self, loss: torch.Tensor):
        """Perform gradient update

        Perform controlled gradient updates for any loss accounting for e.g. gradient clipping.
        Should be called by any variant of self.observe(...).

        Args:
            loss (torch.Tensor): Some loss of the continual learner requiring gradient updates & backprop.
        """
        self.scaler.scale(loss).backward()

        grad_clip_norm = self.args.experiment.optimizer.clip_grad_norm
        if grad_clip_norm > 0:
            self.scaler.unscale_(self.opt)
            gradient_norm_clipper = lambda params: (
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
                if grad_clip_norm > 0
                else lambda x: x
            )
            # We clip potential non-backbone parameters that are optimized.
            optim_parameters = [x["params"] for x in self.opt.param_groups]
            optim_parameters = [x for y in optim_parameters for x in y]
            _ = gradient_norm_clipper(optim_parameters)
            # Now we make sure to also clip all relevant base backbone parameters.
            _ = gradient_norm_clipper(self.backbone.parameters())
        self.scaler.step(self.opt)
        self.scaler.update()

    def observe(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        aux_inputs: torch.Tensor,
        experiment: experiment_lib.PredefinedSequenceExperiment = None,
        texts: List[str] = None,
        **kwargs,
    ) -> float:
        """Default observation function for every Continual Learner.

        Please use self.gradient_update(loss) to update parameters.
        This way, gradient clipping and scaling is automatically included!

        Args:
            images (torch.Tensor): _description_
            targets (torch.Tensor): _description_
            aux_inputs (torch.Tensor): _description_
            experiment (experiment_lib.PredefinedSequenceExperiment, optional): _description_. Defaults to None.
            texts (List[str], optional): _description_. Defaults to None.

        Returns:
            float: Loss value as a float.
        """
        # Please use self.gradient_update(loss) to update parameters.
        # This way, gradient clipping and scaling is automatically included!
        pass

    def forward(
        self,
        images: torch.Tensor,
        texts: List[str] = None,
        experiment: experiment_lib.PredefinedSequenceExperiment = None,
        image_features_only: bool = False,
        **kwargs,
    ) -> dict:
        if self.training and experiment:
            global_task = experiment.global_task      
            self.global_tasks_seen.append(global_task)
            task_level_tasks = ['classification_task', 'classification_seen']
            
            if 'classification' in self.training_mode:
                total_num_classes = experiment.total_num_classes
                
                # For dataset-level changes to training, we perform masking here.
                if self.training_mode == 'classification_dataset':
                    if not hasattr(self, 'current_dataset') or self.current_dataset != experiment.current_experiment:
                        self.current_dataset = experiment.current_experiment                                 
                        self.task_idcs = sorted(list(
                            set(range(total_num_classes)) - set(list(experiment.give_class_indices_of_current_experiment()))))
                
                # For task-specific training protocols, we perform masking here.                            
                if self.training_mode in task_level_tasks and global_task not in self.global_tasks_seen:
                    warm_logit_idcs = experiment.give_class_indices_of_current_task_targets()
                    if self.training_mode == 'classification_task':
                        self.task_mask = torch.ones(total_num_classes)
                    if self.training_mode == 'classification_seen' and not hasattr(self, 'task_mask'):
                        self.task_mask = torch.ones(total_num_classes)
                    self.task_mask[warm_logit_idcs] = 0
                    self.task_idcs = torch.where(self.task_mask)[0]

        # Compute image features.
        contextmanager = (
            torch.no_grad if self.freeze_features else contextlib.nullcontext
        )
        with contextmanager():
            self.backbone_hook_dict["features"] = self.backbone(images)

        if image_features_only:
            return self.backbone_hook_dict["features"]

        # Allow the head to leverage either just the vision features (e.g. simple Linear/MLP) or anything else in addition.
        pass_args = {
            key: item
            for key, item in kwargs.items()
            if key not in self.backbone_hook_dict
        }
        head_out = self.head(texts=texts, **self.backbone_hook_dict, **pass_args)

        # Include the option to mask out non-task-relevant logits during training.
        if (
            self.training
            and self.training_mode in task_level_tasks
            and hasattr(self, "task_mask")
            and len(self.task_idcs) > 0
        ):
            head_out["logits"][:, self.task_idcs] = -float("inf")

        # During training and evaluation, we return all the backbone & head outputs.
        return {**self.backbone_hook_dict, **head_out}

    def end_observe(self) -> None:
        self.scheduler.step()

    @property
    def base_checkpoint(self):
        base_ckpt_2_return = {"self": self.state_dict()}
        if self.args.log.checkpoint_full:
            base_ckpt_2_return["checkpoint_storage"] = self.checkpoint_storage
        return base_ckpt_2_return
    
    @property
    def checkpoint(self):
        return self.base_checkpoint

    def base_checkpoint_loading(self, state_dict):
        self.load_state_dict(state_dict["self"])
        if self.args.log.checkpoint_full:
            self.checkpoint_storage = state_dict["checkpoint_storage"]

    def load_from_checkpoint(self, state_dict):
        self.base_checkpoint_loading(state_dict)