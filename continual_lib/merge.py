import copy
from abc import ABC, abstractmethod

import numpy as np
import math
import torch
from omegaconf import DictConfig, OmegaConf

from collections import OrderedDict
import termcolor
import wandb


################# Utilities #################
def create_merge_instance(args: DictConfig):
    """Creates and returns an instance of the merge class based on the provided configuration.

    Args:
        args (dict): A configuration dictionary containing merge technique details.

    Returns:
        BaseMerge: An instance of the corresponding merge class.

    Raises:
        KeyError: If the provided method does not match any known merge method.
        ValueError: If required configuration parameters are missing.
    """
    method_to_class = {
        "interpolation": Interpolation,
        "slerp": SLERP,
        "ties": TIES,
        "dare_ties": DARETIES,
        "breadcrumbs_ties": BreadcrumbsTIES,
        "task_arithmetic": TaskArithmetic,
        "model_stock": ModelStock,
        "magmax": MagMax,
        "tall_masks": TallMasks,
        "random": Random,
    }
    args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    if "method" not in args:
        raise ValueError("Merge configuration must include a 'method' key.")

    method = args["method"]
    if method not in method_to_class:
        raise KeyError(f"Unknown merge method: {method}")

    # Extract method-specific parameters if they exist
    if method in args:
        method_args = args[method]
    else:
        method_args = {}

    method_agnostic_args = {
        a: args[a] for a in args if a not in ["method"] + list(method_to_class.keys())
    }

    return method_to_class[method](**{**method_args, **method_agnostic_args})


def compute_dots(state_dict1, state_dict2):
    """Compute the dot product between two dictionaries of model weights."""
    dots = {
        key: torch.sum(state_dict1[key] * state_dict2[key])
        for key in state_dict1.keys()
    }
    return {k: v.item() for k, v in dots.items()}


################# Base Classes #################
class BaseMerge(ABC):
    """Abstract base class for merging multiple model state dictionaries."""

    def __init__(self):
        pass

    def __call__(self, state_dicts, zero_shot=None):
        """Validate and merge a list of state dictionaries.

        Args:
            state_dicts (list): List of model state dictionaries.

        Returns:
            dict: The merged state dictionary.

        Raises:
            ValueError: If the state dictionaries have unequal keys.
        """
        self.validate_state_dicts(state_dicts)
        return self.merge(state_dicts, zero_shot)

    @abstractmethod
    def merge(self, state_dicts, zero_shot=None):
        raise NotImplementedError("Merge method not implemented.")

    def validate_state_dicts(self, state_dicts):
        """Check if all dictionaries have the same keys.

        Args:
            state_dicts (list): List of model state dictionaries.

        Raises:
            ValueError: If the dictionaries have unequal keys.
        """
        keys = set(state_dicts[0].keys())

        for state_dict in state_dicts:
            if set(state_dict.keys()) != keys:
                raise ValueError("Cannot merge state dictionaries with unequal keys.")

    def move_state_dicts_to_cpu(self, state_dicts):
        """Move the state dictionaries to the CPU."""
        return [
            {k: v.cpu() for k, v in state_dict.items()} for state_dict in state_dicts
        ]

    def set_weight_coefficients(self, weight_coefficients):
        pass


class BaseWeightedMerge(BaseMerge):
    """Abstract base class for merging multiple model state dictionaries with weighted coefficients."""

    def __init__(self, weight_coefficients=None):
        """Initialize a weighted merge.

        Args:
            weight_coefficients (list of float): The list of weight coefficients.

        Raises:
            TypeError: If `weight_coefficients` is not a list of floats.
        """
        super().__init__()
        self.weight_coefficients = weight_coefficients
        self.rebalance_weights = self.weight_coefficients==None

    def set_weight_coefficients(self, weight_coefficients):
        self.weight_coefficients = weight_coefficients
        self.rebalance_weights = False

    def _adjust_and_validate_weights(self, state_dicts):
        """Adjust and validate the weight coefficients.

        Args:
            state_dicts (list): List of model state dictionaries.

        Raises:
            ValueError: If the number of state dictionaries and weight coefficients are not the same.
        """
        if self.weight_coefficients is None or self.rebalance_weights:
            self.weight_coefficients = [1 / len(state_dicts)] * len(state_dicts)

        if not isinstance(self.weight_coefficients, list):
            raise TypeError("Weight_coefficients must be a list of floats.")

        if len(state_dicts) == len(self.weight_coefficients) + 1:
            raise ValueError(
                f"Weight coefficients and state_dicts are not of same size, weight coefficients are of size {len(self.weight_coefficients)} while state_dicts are of size {len(self.state_dicts)}"
            )

        if abs(sum(self.weight_coefficients) - 1.0) >= 1e-6:
            raise ValueError(
                f"Weight coefficients must sum to 1. Got {self.weight_coefficients}."
            )
        if len(state_dicts) != len(self.weight_coefficients):
            raise ValueError(
                "The number of state dictionaries and weight coefficients must be the same."
            )


class BaseTaskVectorMerge(BaseMerge):
    def __init__(self, scaling_factor=1):
        super().__init__()
        self.scaling_factor = scaling_factor

    def state_dict_to_vector(self, state_dict, remove_keys):
        """Convert a state dictionary to a flattened vector.

        Args:
            state_dict (dict): The state dictionary to convert.
            remove_keys (list): Keys to remove from the state dictionary before conversion.

        Returns:
            torch.Tensor: A flattened vector representation of the state dictionary.
        """
        shared_state_dict = {
            k: v for k, v in state_dict.items() if k not in remove_keys
        }
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for value in shared_state_dict.values()]
        )

    def vector_to_state_dict(self, vector, state_dict, remove_keys):
        """Convert a flattened vector back to a state dictionary.

        Args:
            vector (torch.Tensor): The flattened vector to convert.
            state_dict (dict): The original state dictionary to use as a reference.
            remove_keys (list): Keys that were removed during the flattening process.

        Returns:
            dict: The reconstructed state dictionary.
        """
        reference_dict = {k: v for k, v in state_dict.items() if k not in remove_keys}

        torch.nn.utils.vector_to_parameters(vector, reference_dict.values())

        if "transformer.shared.weight" in reference_dict:
            shared_weight = reference_dict["transformer.shared.weight"]
            for key in remove_keys:
                reference_dict[key] = shared_weight

        return reference_dict


class BaseLinesMerge(ABC):
    """Abstract base class for merges that use lines scaling."""

    def __init__(self, apply_lines=False, lines_params=None):
        if lines_params is None:
            lines_params = (0.5, 0.5, True, "linear")

        if len(lines_params) != 4:
            raise ValueError(
                "You have passed a list of size != 4 to lines_params, please check."
            )

        self.apply_lines = apply_lines
        (
            self.lines_alpha,
            self.lines_beta,
            self.apply_lines_with_weight_coefficients,
            self.lines_scaling_type,
        ) = lines_params
        self.depth = None  # depth of the network (required for lines merging)

    def _infer_depth(self, state_dict):
        """Infer the depth of the network (required for Lines merging)."""
        if self.depth is not None:
            return

        template = "transformer.resblocks."
        max_depth = -1

        for key in state_dict:
            if template in key:
                first_dot_pos = key.index(template) + len(template)
                second_dot_pos = key.index(".", first_dot_pos)
                dep = int(key[first_dot_pos:second_dot_pos])
                max_depth = max(max_depth, dep)

        if max_depth == -1:
            raise ValueError(
                "Unable to parse the keys correctly to find network depth, are you sure you're using a transformer-based model from pytorch/open_clip?"
            )

        self.depth = max_depth + 1

    def _get_lines_weights(self, state_dict, remove_keys):
        """Get flat vector of weights for lines scaling of task vectors."""
        shared_state_dict = {
            k: v for k, v in state_dict.items() if k not in remove_keys
        }

        self._infer_depth(state_dict)
        key_blocks = [f".resblocks.{i}." for i in range(self.depth)]

        scaling_factors = {}
        for key in shared_state_dict.keys():
            for layer, block in enumerate(key_blocks):
                if block in key:
                    if self.lines_scaling_type == "linear":
                        scaling_factors[key] = self.lines_alpha + self.lines_beta * (
                            layer / (self.depth - 1)
                        )
                    elif self.lines_scaling_type == "quadratic":
                        scaling_factors[key] = (
                            self.lines_alpha
                            + self.lines_beta * (layer / (self.depth - 1)) ** 2
                        )
                    elif self.lines_scaling_type == "sqrt":
                        scaling_factors[key] = (
                            self.lines_alpha
                            + self.lines_beta * (layer / (self.depth - 1)) ** 0.5
                        )
                    break

        for key in shared_state_dict.keys():
            if key not in scaling_factors:
                scaling_factors[key] = self.lines_alpha

        lines_weights = torch.nn.utils.parameters_to_vector(
            [
                torch.ones_like(value.reshape(-1)) * scaling_factors[key]
                for key, value in shared_state_dict.items()
            ]
        )

        return lines_weights


################# Method Classes #################
class Interpolation(BaseLinesMerge, BaseWeightedMerge):
    """Interpolation merging technique for multiple model state dictionaries."""

    def __init__(self, weight_coefficients=None, apply_lines=False, lines_params=None):
        """Initialize an interpolation merge.

        Args:
            weight_coefficients (list of float): The list of interpolation weight coefficients.
            apply_lines (bool): Whether to apply lines scaling when merging or not.
            lines_params (list): Contains the alpha, beta, apply_lines_with_weight_coefficients and scaling_type params for lines merging.
                alpha: offset for lines merge
                beta: scale factor for the normalized-layer-weighted scales
                apply_lines_with_weight_coefficients: if true, uses lines weights multiplied by the weight coefficients, else directly uses line weights for scaling and ignores weight coefficients
                scaling_type: can be linear, quadratic or sqrt, denotes the scaling type for the normalized layer weights

        Raises:
            TypeError: If `weight_coefficients` is not a list of floats.
            ValueError: If `lines_params` does not have 4 elements.
        """
        BaseWeightedMerge.__init__(self, weight_coefficients)
        BaseLinesMerge.__init__(self, apply_lines, lines_params)

    def merge(self, state_dicts, zero_shot=None):
        """Interpolate the parameters of multiple state dictionaries.

        Each parameter is interpolated based on the corresponding weight in the `weight_coefficients` list.

        Args:
            state_dicts (list): List of model state dictionaries.

        Returns:
            dict: A merged state dictionary with interpolated parameters.

        Raises:
            ValueError: If the weight coefficients do not sum to 1.
            ValueError: If the length of `weight_coefficients` is not equal to the length of `state_dicts`.
            Note: if only one weight is missing, it will be inferred as 1 - sum(self.weight_coefficients).
        """
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)
        self._adjust_and_validate_weights(state_dicts)

        # Apply lines scaling to weights
        if self.apply_lines:

            # First, get the depth of the model
            self._infer_depth(state_dicts[0])

            # Generate the key blocks corresponding to the layers of the model
            key_blocks = [f".resblocks.{i}." for i in range(self.depth)]

            # Next get appropriate scaling factors based on layers (only for those that can be inferred from keys)
            scaling_factors = {}
            for key in state_dicts[0].keys():
                for layer, block in enumerate(key_blocks):
                    if block in key:
                        if self.lines_scaling_type == "linear":
                            scaling_factors[key] = (
                                self.lines_alpha
                                + self.lines_beta * (layer / (self.depth - 1))
                            )
                        elif self.lines_scaling_type == "quadratic":
                            scaling_factors[key] = (
                                self.lines_alpha
                                + self.lines_beta * (layer / (self.depth - 1)) ** 2
                            )
                        elif self.lines_scaling_type == "sqrt":
                            scaling_factors[key] = (
                                self.lines_alpha
                                + self.lines_beta * (layer / (self.depth - 1)) ** 0.5
                            )
                        break

            termcolor.cprint(
                "Using lines scaling factors with min-scale={} and max-scale={}".format(
                    min(list(scaling_factors.values())),
                    max(list(scaling_factors.values())),
                ),
                "blue",
                attrs=[],
            )

            if self.apply_lines_with_weight_coefficients:
                return {
                    key: sum(
                        w * state_dict[key] * scaling_factors.get(key, self.lines_alpha)
                        for w, state_dict in zip(self.weight_coefficients, state_dicts)
                    )
                    for key in state_dicts[0].keys()
                }

            else:
                return {
                    key: sum(
                        state_dict[key] * scaling_factors.get(key, self.lines_alpha)
                        for state_dict in state_dicts
                    )
                    for key in state_dicts[0].keys()
                }

        # Apply normal scaling for interpolation
        else:

            print('coeffs just before merging', self.weight_coefficients)

            return {
                key: sum(
                    w * state_dict[key]
                    for w, state_dict in zip(self.weight_coefficients, state_dicts)
                )
                for key in state_dicts[0].keys()
            }


class SLERP(BaseLinesMerge, BaseWeightedMerge):
    """Spherical linear interpolation merging technique for two model state dictionaries."""

    def __init__(self, weight_coefficients, apply_lines=False, lines_params=None):
        """Initialize a SLERP merge.

        Args:
            weight_coefficient (float): The weight coefficient for the interpolation.
            apply_lines (bool): Whether to apply lines scaling when merging or not.
            lines_params (list): Contains the alpha, beta, apply_lines_with_weight_coefficients and scaling_type params for lines merging.
                alpha: offset for lines merge
                beta: scale factor for the normalized-layer-weighted scales
                apply_lines_with_weight_coefficients: if true, uses lines weights multiplied by the weight coefficients, else directly uses line weights for scaling and ignores weight coefficients
                scaling_type: can be linear, quadratic or sqrt, denotes the scaling type for the normalized layer weights
        """
        BaseWeightedMerge.__init__(self, weight_coefficients)
        BaseLinesMerge.__init__(self, apply_lines, lines_params)

    def merge(self, state_dicts, zero_shot=None):
        """Merge two state dictionaries using spherical linear interpolation.

        Args:
            state_dicts (list): List of model state dictionaries.

        Returns:
            dict: The merged state dictionary.
        """
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)
        self._adjust_and_validate_weights(state_dicts)

        if len(self.weight_coefficients) not in [1, 2]:
            raise ValueError("SLERP only supports merging two state dictionaries.")

        return {
            key: self.slerp(state_dicts[0][key], state_dicts[1][key])
            for key in state_dicts[0].keys()
        }

    def slerp(self, v0, v1, DOT_THRESHOLD=0.9995, eps=1e-8):
        """Spherical linear interpolation between two vectors."""
        v0_copy = copy.deepcopy(v0)
        v1_copy = copy.deepcopy(v1)

        v0 = self.normalize(v0, eps)
        v1 = self.normalize(v1, eps)

        dot_np = (v0 * v1).sum().item()

        if np.abs(dot_np) > DOT_THRESHOLD:
            return self.lerp(v0_copy, v1_copy)

        omega = np.arccos(dot_np)
        v0_mul = float(
            np.sin(omega * (1 - self.weight_coefficients[0])) / np.sin(omega)
        )
        v1_mul = float(np.sin(omega * self.weight_coefficients[0]) / np.sin(omega))

        return v0_mul * v0_copy + v1_mul * v1_copy

    @staticmethod
    def normalize(v, eps):
        """Normalize a vector to unit length."""
        norm = v.norm()
        if norm > eps:
            return v / norm
        return v

    def lerp(self, v0, v1):
        """Linear interpolation between two vectors."""
        return (1 - self.weight_coefficients[0]) * v0 + self.weight_coefficients[0] * v1


class MagMax(BaseTaskVectorMerge):
    """MagMax merging technique - selects maximum magnitude values from task vectors."""

    def __init__(self, scaling_factor=1, apply_lines=False, lines_params=None):
        """Initialize a MagMax merge.

        Args:
            scaling_factor (float): Scaling factor to apply to the merged task vector.
            ##### lines params are ignored in this method.
            apply_lines (bool): Whether to apply lines scaling when merging or not.
            lines_params (list): Contains the alpha, beta, apply_lines_with_weight_coefficients and scaling_type params for lines merging.
                alpha: offset for lines merge
                beta: scale factor for the normalized-layer-weighted scales
                apply_lines_with_weight_coefficients: if true, uses lines weights multiplied by the weight coefficients, else directly uses line weights for scaling and ignores weight coefficients
                scaling_type: can be linear, quadratic or sqrt, denotes the scaling type for the normalized layer weights
        """
        super().__init__(scaling_factor)

    def merge(self, state_dicts, zero_shot=None):
        """Merge multiple state dictionaries by selecting maximum magnitude values.

        Args:
            state_dicts (list): List of model state dictionaries of the fine-tuned models.
            zero_shot (dict): Zero-shot state dictionary.

        Returns:
            dict: The merged state dictionary.
        """
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)

        remove_keys = [
            "transformer.encoder.embed_tokens.weight",
            "transformer.decoder.embed_tokens.weight",
        ]

        ft_vectors = torch.vstack(
            [
                self.state_dict_to_vector(state_dict, remove_keys)
                for state_dict in state_dicts
            ]
        )
        zs_vector = self.state_dict_to_vector(zero_shot, remove_keys)
        task_vectors = ft_vectors - zs_vector

        # For each parameter, select the value with the largest magnitude
        magnitudes = task_vectors.abs()
        max_magnitude_indices = magnitudes.argmax(dim=0)
        merged_task_vectors = task_vectors[
            max_magnitude_indices, torch.arange(task_vectors.shape[1])
        ]

        merged_vector = zs_vector + self.scaling_factor * merged_task_vectors
        return self.vector_to_state_dict(merged_vector, zero_shot, remove_keys)


class TIES(BaseLinesMerge, BaseTaskVectorMerge):
    """TIES merging technique."""

    def __init__(
        self,
        scaling_factor=1,
        prune_percentile=0.2,
        merge_function="mean",
        apply_lines=False,
        lines_params=None,
    ):
        BaseLinesMerge.__init__(self, apply_lines, lines_params)
        BaseTaskVectorMerge.__init__(self, scaling_factor)
        self.prune_percentile = prune_percentile
        self.merge_function = merge_function

    def merge(self, state_dicts, zero_shot=None):
        """Merge multiple state dictionaries using the TIES technique.

        Args:
            state_dicts (list): List of model state dictionaries of the fine-tuned models.
            zero_shot (dict): Zero-shot state dictionary.

        Returns:
            dict: The merged state dictionary.
        """
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)

        remove_keys = [
            "transformer.encoder.embed_tokens.weight",
            "transformer.decoder.embed_tokens.weight",
        ]

        ft_vectors = torch.vstack(
            [
                self.state_dict_to_vector(state_dict, remove_keys)
                for state_dict in state_dicts
            ]
        )
        zs_vector = self.state_dict_to_vector(zero_shot, remove_keys)
        task_vectors = ft_vectors - zs_vector
        merged_task_vectors = self.ties_merging(task_vectors)

        if self.apply_lines:
            lw = self._get_lines_weights(zero_shot, remove_keys)
            assert (
                lw.shape == zs_vector.shape == merged_task_vectors.shape
            ), "lines-scaling-weights, zs-vector and merged-task-vector dont have the same shape!"
            assert (
                lw.device == zs_vector.device == merged_task_vectors.device
            ), "lines-scaling-weights, zs-vector and merged-task-vector arent on the same device!"
            merged_vector = zs_vector + lw * merged_task_vectors
        else:
            merged_vector = zs_vector + self.scaling_factor * merged_task_vectors
        return self.vector_to_state_dict(merged_vector, zero_shot, remove_keys)

    def ties_merging(self, task_vectors):
        """Perform TIES merging on flattened task checkpoints.

        Args:
            task_vectors (torch.Tensor): Stacked task vectors.

        Returns:
            torch.Tensor: Merged task vector.
        """
        task_vectors = self.sparsify(task_vectors)
        signs = self.resolve_sign(task_vectors)
        return self.disjoint_merge(task_vectors, signs)

    def sparsify(self, task_vectors):
        """Apply a top-k mask to the input tensor.

        Args:
            task_vectors (torch.Tensor): Stacked task vectors.

        Returns:
            torch.Tensor: Masked tensor.
        """
        original_shape = task_vectors.shape

        if task_vectors.dim() == 1:
            task_vectors = task_vectors.unsqueeze(0)

        num_elements = task_vectors.shape[1]
        k = int(num_elements * self.prune_percentile)
        kth_values, _ = task_vectors.abs().kthvalue(k, dim=1, keepdim=True)
        mask = task_vectors.abs() >= kth_values
        mask = (
            mask.squeeze() if original_shape == task_vectors.squeeze().shape else mask
        )
        return task_vectors * mask

    def resolve_sign(self, tensor):
        """Resolve the sign of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with resolved signs.
        """
        sign_to_mult = torch.sign(tensor.sum(dim=0))
        sign_to_mult[sign_to_mult == 0] = torch.sign(sign_to_mult.sum())
        return sign_to_mult

    def disjoint_merge(self, tensor, signs):
        """Perform disjoint merging on the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            signs (torch.Tensor): Tensor with resolved signs.

        Returns:
            torch.Tensor: Merged tensor.

        Raises:
            ValueError: If an undefined merge method is provided.
        """
        rows_to_keep = torch.where(signs.unsqueeze(0) > 0, tensor > 0, tensor < 0)
        selected_entries = tensor * rows_to_keep

        if self.merge_function == "mean":
            non_zero_counts = (selected_entries != 0).sum(dim=0).float()
            disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
                non_zero_counts, min=1
            )
        elif self.merge_function == "sum":
            disjoint_aggs = torch.sum(selected_entries, dim=0)
        elif self.merge_function == "max":
            disjoint_aggs = selected_entries.abs().max(dim=0)[0]
            disjoint_aggs *= signs
        else:
            raise ValueError(f"Merge method {self.merge_function} is not defined.")

        return disjoint_aggs


class DARETIES(TIES):
    """DARE-TIES merging technique."""

    def __init__(
        self,
        scaling_factor=1,
        prune_percentile=0.2,
        merge_function="mean",
        apply_lines=False,
        lines_params=None,
    ):
        super().__init__(
            scaling_factor, prune_percentile, merge_function, apply_lines, lines_params
        )

    def sparsify(self, task_vectors):
        """Apply a random mask to the input tensor.

        Args:
            task_vectors (torch.Tensor): Stacked task vectors.

        Returns:
            torch.Tensor: Masked tensor.
        """
        original_shape = task_vectors.shape

        if task_vectors.dim() == 1:
            task_vectors = task_vectors.unsqueeze(0)

        num_elements = task_vectors.shape[1]
        k = int(num_elements * self.prune_percentile)
        random_values = torch.rand_like(task_vectors)
        threshold_values, _ = random_values.kthvalue(k, dim=1, keepdim=True)
        mask = random_values >= threshold_values
        mask = (
            mask.squeeze() if original_shape == task_vectors.squeeze().shape else mask
        )
        return task_vectors * mask / (1 - self.prune_percentile)


class BreadcrumbsTIES(TIES):
    """Breadcrumbs-TIES merging technique."""

    def __init__(
        self,
        scaling_factor=1,
        top_prune_percentile=0.2,
        prune_percentile=0.2,
        merge_function="sum",
        apply_lines=False,
        lines_params=None,
    ):
        super().__init__(
            scaling_factor, prune_percentile, merge_function, apply_lines, lines_params
        )
        self.top_prune_percentile = top_prune_percentile

    def sparsify(self, task_vectors):
        """Remove elements based on the bottom and top percentiles of magnitudes along the last dimension.

        Args:
            task_vectors (torch.Tensor): Stacked task vectors.

        Returns:
            torch.Tensor: Masked tensor.
        """
        original_shape = task_vectors.shape
        if task_vectors.dim() == 1:
            task_vectors = task_vectors.unsqueeze(0)
        out = torch.stack(
            [
                self.filter_tensor_nd_magnitude(task_vector)
                for task_vector in task_vectors
            ],
            dim=0,
        )
        assert (
            out.shape == original_shape
        ), "Sparsified task vectors are not of the same shape as the original."

        return out

    def filter_tensor_nd_magnitude(self, task_vector):
        """Filters a task vector by removing elements based on the
        bottom and top percentiles of magnitudes along the last dimension.

        Args:
            task_vector (torch.Tensor): The input tensor of arbitrary dimension.

        Returns:
            torch.Tensor: The filtered tensor with masked values set to zero.
        """
        original_shape = task_vector.shape
        if task_vector.ndim == 0:
            return task_vector

        last_dim_size = task_vector.shape[-1]
        reshaped_x = task_vector.view(-1, last_dim_size)
        magnitudes = reshaped_x.abs()

        # Calculate the thresholds
        top_k = int(self.top_prune_percentile * last_dim_size)
        bottom_k = int(self.prune_percentile * last_dim_size)
        if top_k + bottom_k >= last_dim_size:
            raise ValueError(
                f"Invalid prune percentiles: {self.top_prune_percentile} and {self.prune_percentile} "
                f"would remove all {last_dim_size} values"
            )

        # Determine threshold values based on magnitudes
        sorted_magnitudes, _ = torch.sort(magnitudes, dim=1)
        min_thresholds = (
            sorted_magnitudes[:, bottom_k] if bottom_k > 0 else sorted_magnitudes[:, 0]
        )
        max_thresholds = (
            sorted_magnitudes[:, -top_k - 1] if top_k > 0 else sorted_magnitudes[:, -1]
        )
        min_thresholds = min_thresholds.unsqueeze(1).expand(-1, last_dim_size)
        max_thresholds = max_thresholds.unsqueeze(1).expand(-1, last_dim_size)

        mask = (magnitudes >= min_thresholds) & (magnitudes <= max_thresholds)
        filtered_reshaped_x = reshaped_x * mask.float()
        return filtered_reshaped_x.view(original_shape)


class TaskArithmetic(BaseLinesMerge, BaseTaskVectorMerge):
    """Task-Arithmetic merging technique."""

    def __init__(self, scaling_factor=1, apply_lines=False, lines_params=None):
        BaseLinesMerge.__init__(self, apply_lines, lines_params)
        BaseTaskVectorMerge.__init__(self, scaling_factor)

    def merge(self, state_dicts, zero_shot=None):
        """Merge multiple state dictionaries using the Task-Arithmetic technique.

        Args:
            state_dicts (list): List of model state dictionaries of the fine-tuned models.
            zero_shot (dict): Zero-shot state dictionary.

        Returns:
            dict: The merged state dictionary.
        """
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)

        remove_keys = [
            "transformer.encoder.embed_tokens.weight",
            "transformer.decoder.embed_tokens.weight",
        ]

        ft_vectors = torch.vstack(
            [
                self.state_dict_to_vector(state_dict, remove_keys)
                for state_dict in state_dicts
            ]
        )
        zs_vector = self.state_dict_to_vector(zero_shot, remove_keys)
        task_vectors = ft_vectors - zs_vector
        merged_task_vectors = self.tv_merge(task_vectors)

        if self.apply_lines:
            lw = self._get_lines_weights(zero_shot, remove_keys)
            assert (
                lw.shape == zs_vector.shape == merged_task_vectors.shape
            ), "lines-scaling-weights, zs-vector and merged-task-vector dont have the same shape!"
            assert (
                lw.device == zs_vector.device == merged_task_vectors.device
            ), "lines-scaling-weights, zs-vector and merged-task-vector arent on the same device!"
            merged_vector = zs_vector + lw * merged_task_vectors
        else:
            merged_vector = zs_vector + self.scaling_factor * merged_task_vectors
        return self.vector_to_state_dict(merged_vector, zero_shot, remove_keys)

    def tv_merge(self, task_vectors):
        """Perform Task=Arithmetic merging on flattened task checkpoints.

        Args:
            task_vectors (torch.Tensor): Flattened task checkpoints.

        Returns:
            torch.Tensor: Merged task vector.
        """
        # TODO (vishaal): for now just implementing the sum as the aggregation function
        # see here for other potential aggregation functions: https://github.com/prateeky2806/ties-merging/blob/44e7891fc84f3de7e4caa52664cd864ca3715e91/src/utils/merge_utils.py#L376
        return torch.sum(task_vectors, dim=0)


class ModelStock(BaseWeightedMerge):
    """Model-Stock merging technique."""

    def __init__(self, eps=1e-8, apply_lines=False, lines_params=None):
        """Initialize a Model-Stock merge."""
        BaseLinesMerge.__init__(self, apply_lines, lines_params)
        # NOTE: for model-stock, we cannot use lines scaling since the weights are self-inferred
        self.eps = eps

    def merge(self, state_dicts, zero_shot=None):
        """Merge two state dictionaries along with zero-shot state dict using model-stock.

        Args:
            state_dicts (list): List of model state dictionaries.

        Returns:
            dict: The merged state dictionary.
        """
        # TODO (vishaal): For now, only implementing the 2 dict version,
        # can generalize it later to the N dict version if promising.
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)
        if len(state_dicts) not in [2]:
            raise ValueError(
                "Currently, the ModelStock impl only supports two state dictionaries."
            )

        # get angle
        angles = self.compute_angle(state_dicts, zero_shot)

        # get ratio
        ratios = self.compute_ratio(angles)

        # get w12 = (w1 + w2) / 2
        w12 = {}
        for key in state_dicts[0].keys():
            w12[key] = 0.5 * (state_dicts[0][key] + state_dicts[1][key])

        out_dict = {}
        for key, r in ratios.items():
            out_dict[key] = w12[key] * r + zero_shot[key] * (1.0 - r)

        return out_dict

    def compute_angle(self, state_dicts, zero_shot):
        """Compute angle theta for model stock ratio computation"""

        # TODO (vishaal): for now we only support two models in the merge
        assert (
            len(state_dicts) == 2
        ), "For now, ModelStock only supports merging with two state dicts."

        return_dict = OrderedDict()

        for key in zero_shot:
            state_dict_1_val = state_dicts[0][key]
            state_dict_2_val = state_dicts[1][key]
            zs_val = zero_shot[key]

            assert (
                state_dict_1_val.shape == state_dict_2_val.shape == zs_val.shape
            ), "Mismatched shapes while computing angle for ModelStock merging."

            vector1 = (state_dict_1_val - zs_val).clone().detach().float()
            vector2 = (state_dict_2_val - zs_val).clone().detach().float()

            cosine_val = torch.sum(vector1 * vector2) / (
                math.sqrt(torch.sum(vector1**2) * torch.sum(vector2**2)) + self.eps
            )
            cosine_val = torch.clamp(
                cosine_val, min=-1.0, max=1.0
            )  # To prevent nan from acos
            return_dict[key] = cosine_val.detach().cpu().item()

        return return_dict

    def compute_ratio(self, angle_dict, k=2):
        # TODO (vishaal): for now we only support two models in the merge
        assert k == 2, "For now, ModelStock only supports merging with two state dicts."

        ratio_dict = {}

        for key in angle_dict.keys():
            cosval = angle_dict[key]
            ratio_dict[key] = k * cosval / ((k - 1) * cosval + 1 + self.eps)

        return ratio_dict


class TallMasks(BaseTaskVectorMerge):
    """TallMasks merging technique that uses consensus masking."""

    def __init__(self, scaling_factor=1, pruning_threshold=0, tall_mask_lambda=0.4, apply_lines=False, lines_params=None):
        """Initialize a TallMasks merge.

        Args:
            scaling_factor (float): Scaling factor to apply to the merged task vector
            pruning_threshold (int): Minimum number of activated tasks for a parameter to be preserved
            tall_mask_lambda (float): Hyperparameter lambda for generating TALL masks
        """
        super().__init__(scaling_factor)
        self.pruning_threshold = pruning_threshold
        self.tall_mask_lambda = tall_mask_lambda

    def merge(self, state_dicts, zero_shot=None):
        """Merge multiple state dictionaries using the TallMasks technique.

        Args:
            state_dicts (list): List of model state dictionaries of the fine-tuned models
            zero_shot (dict): Zero-shot state dictionary

        Returns:
            dict: The merged state dictionary
        """
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)

        remove_keys = [
            "transformer.encoder.embed_tokens.weight",
            "transformer.decoder.embed_tokens.weight",
        ]

        # Convert state dicts to vectors
        ft_vectors = torch.vstack(
            [
                self.state_dict_to_vector(state_dict, remove_keys)
                for state_dict in state_dicts
            ]
        )
        zs_vector = self.state_dict_to_vector(zero_shot, remove_keys)

        consensus_mask = self._generate_consensus_mask(ft_vectors, zs_vector)

        # Calculate and mask the merged task vector
        merged_tv = (ft_vectors - zs_vector).sum(0)
        masked_tv = merged_tv * consensus_mask
        merged_vector = zs_vector + self.scaling_factor * masked_tv

        return self.vector_to_state_dict(merged_vector, zero_shot, remove_keys)

    def _generate_consensus_mask(self, ft_vectors, zs_vector):
        """Generate consensus mask by filtering out least-used parameters.

        Args:
            ft_vectors (torch.Tensor): Individual theta_t (fine-tuned weights)
            zs_vector (torch.Tensor): theta_0 (pre-trained weights)

        Returns:
            torch.Tensor: Consensus mask as vector (boolean)
        """
        tall_masks = self._generate_task_masks(ft_vectors, zs_vector)

        # Count activated tasks per parameter
        activation_counts = tall_masks.float().sum(0)

        # Generate consensus mask based on activation threshold
        consensus_mask = activation_counts >= self.pruning_threshold

        wandb.log({"consensus_mask_sparsity": consensus_mask.float().mean()})

        return consensus_mask

    def _generate_task_masks(self, ft_vectors, zs_vector):
        """Generate task-specific TALL masks.

        Args:
            flat_ft (torch.Tensor): Individual theta_t (fine-tuned weights)
            flat_ptm (torch.Tensor): theta_0 (pre-trained weights)

        Returns:
            torch.Tensor: Generated TALL masks with the given lambda
        """
        merged_tv = (ft_vectors - zs_vector).sum(0)
        multi_vector = zs_vector + merged_tv

        diff_pt_ft = (zs_vector - ft_vectors).abs()
        diff_multi_ft = (multi_vector - ft_vectors).abs()
        mask = diff_pt_ft > diff_multi_ft * self.tall_mask_lambda

        final_mask = mask.squeeze() if len(mask.shape) > 2 else mask

        return final_mask


class Random(BaseTaskVectorMerge):
    """Random merging technique - randomly selects values from task vectors."""

    def __init__(self, scaling_factor=1, apply_lines=False, lines_params=None):
        """Initialize a Random merge.

        Args:
            scaling_factor (float): Scaling factor to apply to the merged task vector.
        """
        super().__init__(scaling_factor)

    def merge(self, state_dicts, zero_shot=None):
        """Merge multiple state dictionaries by randomly selecting values.

        Args:
            state_dicts (list): List of model state dictionaries of the fine-tuned models.
            zero_shot (dict): Zero-shot state dictionary.

        Returns:
            dict: The merged state dictionary.
        """
        state_dicts = self.move_state_dicts_to_cpu(state_dicts)

        remove_keys = [
            "transformer.encoder.embed_tokens.weight",
            "transformer.decoder.embed_tokens.weight",
        ]

        ft_vectors = torch.vstack(
            [
                self.state_dict_to_vector(state_dict, remove_keys)
                for state_dict in state_dicts
            ]
        )
        zs_vector = self.state_dict_to_vector(zero_shot, remove_keys)
        task_vectors = ft_vectors - zs_vector

        # For each parameter, randomly select a value
        random_indices = torch.randint(0, task_vectors.shape[0], (task_vectors.shape[1],))
        merged_task_vectors = task_vectors[random_indices, torch.arange(task_vectors.shape[1])]

        merged_vector = zs_vector + self.scaling_factor * merged_task_vectors
        return self.vector_to_state_dict(merged_vector, zero_shot, remove_keys)
