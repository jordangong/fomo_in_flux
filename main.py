import os
import pathlib

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
import termcolor
import torch

import backbones
import continual_lib
import data_lib
import experiment_lib
import utils.conf
import utils.evaluation
import utils.training
import utils.sequence_handling

OmegaConf.register_new_resolver("eval", eval)


def calculate_individual_batch_sizes(total_batch_size, data_mixture):
    """
    Calculate individual batch sizes for each data mixture component.

    Args:
        total_batch_size (int): The total batch size to be divided.
        data_mixture (dict): A dictionary with mixture components and their ratios.

    Returns:
        dict: A dictionary with the same keys as data_mixture and their respective batch sizes.
    """
    # Ensure mixture weights sum to 1
    mixture_values = list(data_mixture.values())
    assert (
        abs(sum(map(float, mixture_values)) - 1.0) < 1e-6
    ), "Mixture component ratios do not sum to 1"

    # Calculate batch sizes based on the given ratios
    batch_sizes = {
        key: int(total_batch_size * ratio) for key, ratio in data_mixture.items()
    }

    # Calculate the total batch size from the computed batch sizes
    computed_total = sum(batch_sizes.values())

    # Adjust the batch sizes if necessary to ensure the total sum matches the total_batch_size
    if computed_total < total_batch_size:
        # Distribute the remaining batch size
        remaining = total_batch_size - computed_total
        for key in data_mixture:
            if remaining == 0:
                break
            batch_sizes[key] += 1
            remaining -= 1

    return batch_sizes


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig) -> None:
    print("\n")

    ########### Default value adjustments.
    # Backward compatibility.
    if "freeze_visual" in args.experiment.backbone:
        raise Exception(
            "Using deprecated <freeze_visual>. Please switch to <freeze_features>."
        )
    if "freeze_semantics" in args.experiment.backbone:
        raise Exception(
            "Using deprecated <freeze_semantics>. Please switch to <freeze_head>."
        )
    with open_dict(args):
        args.experiment.backbone.freeze_visual = (
            args.experiment.backbone.freeze_features
        )
        args.experiment.backbone.freeze_semantics = args.experiment.backbone.freeze_head

    # Check if training mode is one of the options allowed.
    assert_str = f"Training mode {args.experiment.training} not available. Please choose from {continual_lib.ALLOWED_TRAINING}!"
    assert args.experiment.training in continual_lib.ALLOWED_TRAINING, assert_str

    # Check if dataset_incremental training is used. If so, check that dataset.name is defined,
    # dataset.sequence is unset, sequence_reshuffle is off and auto-set task.num.
    if args.experiment.task.dataset_incremental:
        assert_str = (
            f"Using dataset-incremental training. Please do not set dataset.sequence!"
        )
        assert args.experiment.dataset.sequence is None, assert_str
        assert_str = "No sequence reshuffling allowed in dataset-incremental training!"
        assert not args.experiment.dataset.sequence_reshuffle, assert_str
        if not isinstance(args.experiment.dataset.name, ListConfig):
            args.experiment.dataset.name = [args.experiment.dataset.name]
        args.experiment.task.num = len(args.experiment.dataset.name)

    # Set specific number of `gap_samples` based on `eval_every_n_samples` for stability gap studies
    if (
        args.experiment.task.eval_every_n_samples is not None
        and args.experiment.task.n_samples is not None
    ):
        args.experiment.task.gap_samples = list(
            range(
                0,
                args.experiment.task.n_samples,
                args.experiment.task.eval_every_n_samples,
            )
        )

    # If no specific value is given for the batchsize to use at test time, simply use the training batchsize.
    if args.experiment.evaluation.batch_size < 0:
        args.experiment.evaluation.batch_size = args.experiment.task.batch_size

    ### data-mixture values for this instead of having a different buffer bs?
    # If no specific value is given for the batchsize to sample from buffers, simply use the training batchsize.
    if args.experiment.buffer.batch_size < 0:
        args.experiment.buffer.batch_size = args.experiment.task.batch_size
    assert (
        args.experiment.buffer.batch_size <= args.experiment.task.batch_size
    ), "Please ensure that buffer batchsize <= task batchsize!"

    # Check if task sequence is provided.
    utils.sequence_handling.update_args(args)

    # Training-specific arguments.
    if not args.zeroshot_only:
        # Scale learning rate based on batch-size.
        if args.experiment.optimizer.scaled_learning_rate:
            args.experiment.optimizer.lr = (
                args.experiment.optimizer.lr * args.experiment.task.batch_size / 256
            )

        # Only allow full CLIP models in training=contrastive mode!
        clip_models = backbones.clip_models + backbones.openclip_models
        using_clip_model = (
            args.experiment.backbone.name in clip_models
            and args.experiment.backbone.head == "default"
        )
        is_contrastive_training = args.experiment.training == "contrastive"
        if using_clip_model and not is_contrastive_training:
            raise AssertionError(
                "Joint, CLIP-style training of vision and text encoder only allowed "
                "for experiment.training=contrastive!\nCurrently, backbone "
                f"experiment.backbone.name={args.experiment.backbone.name}, "
                f"experiment.backbone.head={args.experiment.backbone.head} "
                f"with experiment.training={args.experiment.training}!"
            )

    ########### Base Script.
    ### Print Summary:
    termcolor.cprint("> Run Arguments.", "white", attrs=["bold"])
    utils.conf.summarize_args(args)

    ### Set default seed.
    if args.experiment.seed is not None:
        utils.conf.set_random_seed(args.experiment.seed, set_backend=args.log.full_replication)
    device = "cuda"

    ### Create per data-mixture batch sizes
    data_mix_batch_sizes = calculate_individual_batch_sizes(
        args.experiment.task.batch_size, args.experiment.task.data_mixture
    )
    with open_dict(args):
        args.experiment.task.update_pool_batch_size = data_mix_batch_sizes["update"]
        args.experiment.task.buffer_pool_batch_size = data_mix_batch_sizes["buffer"]
        args.experiment.task.pretraining_batch_size = data_mix_batch_sizes[
            "pretraining"
        ]

    ### Grab datasets.
    termcolor.cprint("\n> Setting datasets.", "white", attrs=["bold"])
    datasets_dict = data_lib.get_datasets(
        args,
        train_transform=args.experiment.dataset.train_transforms,
        test_transform=args.experiment.dataset.test_transforms,
    )
    data_lib.summarize(args, datasets_dict)

    ### Init experiment handler.
    termcolor.cprint("\n> Setting Experiments.", "white", attrs=["bold"])
    experiment_kwargs = {
        "args": args,
        "train_datasets": datasets_dict["train"],
        "test_datasets": datasets_dict["test"],
        "device": device,
        "task_sequence": args.experiment.dataset.sequence,
        "dataset_names": args.experiment.dataset.name,
    }
    experiment = experiment_lib.PredefinedSequenceExperiment(**experiment_kwargs)
    experiment.summary()

    ### Load vision backbone.
    termcolor.cprint("\n> Loading and setting up model.", "white", attrs=["bold"])
    classnames = [list(x) for x in args.experiment.dataset.classes if x is not None]
    classnames = [x for y in classnames for x in y]
    backbone, head, data_params_updates = backbones.get_backbone_and_head(
        device, args, classnames
    )

    num_params_backbone = sum([params.numel() for params in backbone.parameters()])
    num_params_head = sum([params.numel() for params in head.parameters()])
    print(
        f"Backbone Info:\n - Name: {args.experiment.backbone.name}.\n - Pretraining: {args.experiment.backbone.pretrained}.\n - Num. Parameters: {num_params_backbone}."
    )
    print(
        f"Head Info:\n - Name: {args.experiment.backbone.head}.\n - Num. Parameters: {num_params_head}."
    )
    backbone = torch.nn.DataParallel(backbone, device_ids=args.gpu)
    head = torch.nn.DataParallel(head, device_ids=args.gpu)
    backbone.pretrained = args.experiment.backbone.pretrained
    ### Update experiment class with special requirements for CL.
    # This mainly includes what type of outputs are required (e.g. auxiliary augmentations or unaugmented variants).
    # -> data_req_updates
    # In addition, if the backbone requires changes to the augmentation protocol, this will be updated here:
    # -> data_params_updates
    data_req_updates = {
        "req_aux_inputs": continual_lib.get_class(args).REQ_AUX_INPUTS,
        "req_non_aug_inputs": continual_lib.get_class(args).REQ_NON_AUG_INPUTS,
    }
    if args.experiment.buffer.with_transform:
        # When a method utilizes a buffer & the buffer returns transformed images,
        # we have to request non-augmented input data as well.
        data_req_updates["req_non_aug_inputs"] = True
    if args.experiment.dataset.img_size > 0:
        # Image-size is generally determined by the chosen dataset.
        # However, custom dataset.img_size overwrites this.
        data_params_updates["img_size"] = args.experiment.dataset.img_size
    if args.experiment.dataset.resize > 0:
        # Image resizing is generally determined by the chosen dataset.
        # However, custom dataset.resize overwrites this.
        data_params_updates["resize"] = args.experiment.dataset.resize

    # Update dataset parameters in the experiment.
    experiment.update_dataset_parameters(
        data_req_updates=data_req_updates, data_params_updates=data_params_updates
    )

    ### Set up continual learner.
    termcolor.cprint("\n> Setting Continual Learner.", "white", attrs=["bold"])
    continual_learner = continual_lib.get(
        args,
        backbone,
        head,
        continual_lib.get_loss(args),
        device,
        experiment,
        params=args.continual[args.continual.method],
    )
    # Change [2/2] to handle buffer methods with transformation buffers.
    continual_learner.REQ_NON_AUG_INPUTS = True

    assert_str = f"Continual learner [{args.continual.method}] does not support [{args.experiment.training}] training mode!"
    assert (
        args.experiment.training in continual_learner.SUPPORTED_TRAINING_MODES
    ), assert_str
    print(f"Continual Learner Info:\n - Method: {args.continual.method}")

    ### Set up Evaluator Class Instance.
    termcolor.cprint("\n> Setting Evaluator.", "white", attrs=["bold"])
    if args.log.id is None:
        log_folder = os.path.join(
            args.log.folder,
            f"{args.log.project}_{args.log.group}_{args.log.name}_s-{args.experiment.seed}_{experiment.name}",
        )
    else:
        log_folder = os.path.join(args.log.folder, str(args.log.id))

    print(f"Utilized project folder: {log_folder}.")
    with open_dict(args):
        args.log.log_folder = pathlib.Path(log_folder)

    evaluator = utils.evaluation.Evaluator(
        args,
        experiment,
        device,
        log_folder=log_folder,
        evaluation_only_test_datasets=datasets_dict["eval_only_test"],
    )
    evaluator.update_dataset_parameters(
        data_req_updates=data_req_updates, data_params_updates=data_params_updates
    )
    print(
        "- Task metrics: {}".format(" | ".join(args.experiment.evaluation.task_metrics))
    )
    print(
        "- Total metrics: {}".format(
            " | ".join(args.experiment.evaluation.total_metrics)
        )
    )
    if len(args.experiment.evaluation.additional_datasets):
        print(
            "- EVAL ONLY Datasets: {}".format(
                " | ".join(args.experiment.evaluation.additional_datasets)
            )
        )
    else:
        print("- No EVAL ONLY Datasets used!")

    ### Train (and also evaluate) the continual learner.
    termcolor.cprint(
        "\n\n> Starting Main Adaptation Process.\n", "green", attrs=["bold"]
    )
    utils.training.train(args, continual_learner, experiment, evaluator)


if __name__ == "__main__":
    main()
