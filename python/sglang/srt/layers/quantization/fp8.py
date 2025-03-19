# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    apply_fp8_linear,
    convert_to_channelwise,
    cutlass_fp8_supported,
    per_tensor_dequantize,
    requantize_with_max_scale,
)

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_utils import (
    BlockQuantScaleParameter,
    apply_w8a8_block_fp8_linear,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_hip,
    permute_weight,
    print_warning_once,
    set_weight_attrs,
)

from sglang.srt.layers.activation import SiluAndMul

ACTIVATION_SCHEMES = ["static", "dynamic"]

is_hip_ = is_hip()

logger = logging.getLogger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning(
                "Detected fp8 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports fp8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")
        # Disable marlin for ROCm
        if is_hip_:
            self.use_marlin = False

        self.block_quant = self.quant_config.weight_block_size is not None
        if self.block_quant:
            # Marlin doesn't support block-wise fp8
            self.use_marlin = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if tp_size > 1 and input_size // input_size_per_partition == tp_size:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by collum parallel or enabling merged weights
            if (
                tp_size > 1 and output_size // output_size_per_partition == tp_size
            ) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                assert self.quant_config.activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        if self.block_quant:
            # If ROCm, normalize the weights and scales to e4m3fnuz
            if is_hip_:
                # activation_scheme: dynamic
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=None,
                )
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                layer.weight_scale_inv = torch.nn.Parameter(
                    weight_scale, requires_grad=False
                )
                layer.input_scale = None
            else:
                layer.weight = torch.nn.Parameter(
                    layer.weight.data, requires_grad=False
                )
                layer.weight_scale_inv = torch.nn.Parameter(
                    layer.weight_scale_inv.data, requires_grad=False
                )
            return
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        # If checkpoint not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)

            # If using marlin (w8a16), kernel uses channelwise weights,
            # so extend the weight scales to be channelwise.
            if self.use_marlin:
                assert weight_scale.numel() == 1
                weight_scale = convert_to_channelwise(
                    weight_scale.expand(len(layer.logical_widths)), layer.logical_widths
                )

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

        # If checkpoint is fp8, handle that there are N scales for N
        # shards in a fused module
        else:
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = torch.nn.Parameter(
                    layer.input_scale.data, requires_grad=False
                )
            # If using marlin (w8a16), kernel uses channelwise weights,
            # so extend the weight scales to be channelwise.
            if self.use_marlin:
                weight = layer.weight
                weight_scale = convert_to_channelwise(
                    layer.weight_scale, layer.logical_widths
                )

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            else:
                # Dequant -> Quant with max scale so we can run per tensor.
                weight = layer.weight
                weight_scale = layer.weight_scale

                # If ROCm, normalize the weights and scales to e4m3fnuz
                if is_hip_:
                    weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                        weight=weight,
                        weight_scale=weight_scale,
                        input_scale=layer.input_scale,
                    )
                    if input_scale is not None:
                        layer.input_scale = Parameter(input_scale, requires_grad=False)

                weight_scale, weight = requantize_with_max_scale(
                    weight=weight,
                    weight_scale=weight_scale,
                    logical_widths=layer.logical_widths,
                )

            # Update layer with new values.
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(
                    layer.input_scale.max(), requires_grad=False
                )

        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer)
            # Activations not quantized for marlin.
            del layer.input_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        if self.block_quant:
            return apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=False,
        )


class Fp8MoEMethod:
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
            # Required by collum parallel or enabling merged weights
            if intermediate_size % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                # Required by row parallel
                if intermediate_size % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
        else:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            padding_size,  # Avoid circular import
        )

        # Block quant doesn't need to process weights after loading
        if self.block_quant:
            # If ROCm, normalize the weights and scales to e4m3fnuz
            if is_hip_:
                # activation_scheme: dynamic
                w13_weight, w13_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.w13_weight,
                    weight_scale=layer.w13_weight_scale_inv,
                    input_scale=None,
                )
                w2_weight, w2_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale_inv,
                    input_scale=None,
                )
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale_inv = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                layer.w13_input_scale = None
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale_inv = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
                layer.w2_input_scale = None
            return
        # If checkpoint is fp16 or bfloat16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If ROCm, use float8_e4m3fnuz instead (MI300x HW)
            fp8_dtype = torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts, dtype=torch.float32, device=w13_weight.device
                ),
                requires_grad=False,
            )
            for expert in range(layer.num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

            if is_hip_:
                if get_bool_env_var("CK_MOE"):
                    layer.w13_weight = torch.nn.Parameter(
                        permute_weight(layer.w13_weight.data),
                        requires_grad=False,
                    )
                    torch.cuda.empty_cache()
                    layer.w2_weight = torch.nn.Parameter(
                        permute_weight(layer.w2_weight.data),
                        requires_grad=False,
                    )
                    torch.cuda.empty_cache()
                elif get_bool_env_var("MOE_PADDING"):
                    # If ROCm, apply weight padding (min. Mem channel contention) only if set
                    layer.w13_weight = torch.nn.Parameter(
                        F.pad(layer.w13_weight.data, (0, padding_size), "constant", 0),
                        requires_grad=False,
                    )
                    torch.cuda.empty_cache()
                    layer.w2_weight = torch.nn.Parameter(
                        F.pad(layer.w2_weight.data, (0, padding_size), "constant", 0),
                        requires_grad=False,
                    )
                    torch.cuda.empty_cache()
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                    layer.w2_input_scale
                ):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. "
                    )
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False
                )
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False
                )

            # If ROCm, normalize the weights and scales to e4m3fnuz
            if is_hip_:
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                    )
                )
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False
                    )
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False
                    )
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    layer.w13_weight[expert_id][start : start + shard_size, :], _ = (
                        ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                    )
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

            if is_hip_:
                if get_bool_env_var("CK_MOE"):
                    layer.w13_weight = torch.nn.Parameter(
                        permute_weight(layer.w13_weight.data),
                        requires_grad=False,
                    )
                    torch.cuda.empty_cache()
                    layer.w2_weight = torch.nn.Parameter(
                        permute_weight(layer.w2_weight.data),
                        requires_grad=False,
                    )
                    torch.cuda.empty_cache()
                elif get_bool_env_var("MOE_PADDING"):
                    # If ROCm, apply weight padding (min. Mem channel contention) only if set
                    layer.w13_weight = torch.nn.Parameter(
                        F.pad(layer.w13_weight.data, (0, padding_size), "constant", 0),
                        requires_grad=False,
                    )
                    torch.cuda.empty_cache()
                    layer.w2_weight = torch.nn.Parameter(
                        F.pad(layer.w2_weight.data, (0, padding_size), "constant", 0),
                        requires_grad=False,
                    )
                    torch.cuda.empty_cache()
            return

    def torch_w8a8_block_fp8_moe(self, a, w1, w2, w1_s, w2_s, score, topk, block_shape):
        def native_per_token_group_quant_fp8(
            x, group_size, eps=1e-10, dtype=torch.float8_e4m3fn
        ):
            """Function to perform per-token-group quantization on an input tensor `x` using native torch.

            It converts the tensor values into float8 values and returns the
            quantized tensor along with the scaling factor used for quantization.
            Note that only `torch.float8_e4m3fn` is supported for now.
            """
            assert (
                x.shape[-1] % group_size == 0
            ), "the last dimension of `x` cannot be divisible by `group_size`"
            assert x.is_contiguous(), "`x` is not contiguous"

            finfo = torch.finfo(dtype)
            fp8_min = finfo.min
            fp8_max = finfo.max

            x_ = x.reshape(x.numel() // group_size, group_size)
            amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
            x_s = amax / fp8_max
            x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
            x_q = x_q.reshape(x.shape)
            x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

            return x_q, x_s
        
        def native_w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float16):
            """This function performs matrix multiplication with block-wise quantization using native torch.

            It takes two input tensors `A` and `B` with scales `As` and `Bs`.
            The output is returned in the specified `output_dtype`.
            """

            A = A.to(torch.float32)
            B = B.to(torch.float32)
            assert A.shape[-1] == B.shape[-1]
            assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
            assert len(block_size) == 2
            block_n, block_k = block_size[0], block_size[1]
            assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
            assert A.shape[:-1] == As.shape[:-1]

            M = A.numel() // A.shape[-1]
            N, K = B.shape
            origin_C_shape = A.shape[:-1] + (N,)
            A = A.reshape(M, A.shape[-1])
            As = As.reshape(M, As.shape[-1])
            n_tiles = (N + block_n - 1) // block_n
            k_tiles = (K + block_k - 1) // block_k
            assert n_tiles == Bs.shape[0]
            assert k_tiles == Bs.shape[1]

            C_shape = (M, N)
            C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

            A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
            B_tiles = [
                [
                    B[
                        j * block_n : min((j + 1) * block_n, N),
                        i * block_k : min((i + 1) * block_k, K),
                    ]
                    for i in range(k_tiles)
                ]
                for j in range(n_tiles)
            ]
            C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
            As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

            for i in range(k_tiles):
                for j in range(n_tiles):
                    a = A_tiles[i]
                    b = B_tiles[j][i]
                    c = C_tiles[j]
                    s = As_tiles[i] * Bs[j][i]
                    c[:, :] += torch.matmul(a, b.t()) * s

            C = C.reshape(origin_C_shape).to(output_dtype)
            return C

        """This function performs fused moe with block-wise quantization using native torch."""

        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)

        _, block_k = block_shape[0], block_shape[1]
        a_q, a_s = native_per_token_group_quant_fp8(a, block_k)
        # NOTE(HandH1998): Since "index_cuda" not implemented for 'Float8_e4m3fn', we need to cast `float8`` to `float32``.
        a_q = a_q.to(torch.float32)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                inter_out = native_w8a8_block_fp8_matmul(
                    a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
                )
                act_out = SiluAndMul().forward_native(inter_out)
                act_out_q, act_out_s = native_per_token_group_quant_fp8(act_out, block_k)
                act_out = act_out.to(torch.float32)
                out[mask] = native_w8a8_block_fp8_matmul(
                    act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
                )
        return (
            out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
        ).sum(dim=1)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        '''
        # Expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
        )

        if is_hip_ and get_bool_env_var("CK_MOE"):
            import ater
            from ater.fused_moe import fused_experts_ck

            assert activation == "silu", f"{activation=} is not supported."

            return fused_experts_ck(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                use_fp8_w8a8=True,
                w1_scale=(
                    layer.w13_weight_scale_inv
                    if self.block_quant
                    else layer.w13_weight_scale
                ),
                w2_scale=(
                    layer.w2_weight_scale_inv
                    if self.block_quant
                    else layer.w2_weight_scale
                ),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )

        else:
            # Expert fusion with FP8 quantization
            return fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                use_fp8_w8a8=True,
                w1_scale=(
                    layer.w13_weight_scale_inv
                    if self.block_quant
                    else layer.w13_weight_scale
                ),
                w2_scale=(
                    layer.w2_weight_scale_inv
                    if self.block_quant
                    else layer.w2_weight_scale
                ),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                block_shape=self.quant_config.weight_block_size,
            )
        '''
        
        return self.torch_w8a8_block_fp8_moe(
                a=x, 
                w1=layer.w13_weight, 
                w2=layer.w2_weight, 
                w1_s=(
                    layer.w13_weight_scale_inv
                    if self.block_quant
                    else layer.w13_weight_scale
                ), 
                w2_s=(
                    layer.w2_weight_scale_inv
                    if self.block_quant
                    else layer.w2_weight_scale
                ), 
                score=router_logits, 
                topk=top_k, 
                block_shape=self.quant_config.weight_block_size
            )



class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
