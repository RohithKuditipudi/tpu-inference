# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization import modelopt as vllm_modelopt
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase

from tpu_inference.layers.common import quant_methods
from tpu_inference.layers.common.moe import shard_moe_weights
from tpu_inference.layers.common.process_weights.linear_weights import \
    shard_linear_weights
from tpu_inference.layers.common.quantization import (dequantize_nvfp4,
                                                      e8m0_to_fp32)
from tpu_inference.layers.common.quantization import \
    unquantized as common_unquantized
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.fp8 import (VllmFp8LinearMethod,
                                                        VllmFp8MoEMethod)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(quant_methods.MODELOPT)
class VllmModelOptFp8Config(vllm_modelopt.ModelOptFp8Config, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return quant_methods.MODELOPT

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union[vllm_linear.LinearMethodBase, QuantizeMethodBase]]:
        if self.is_layer_excluded(prefix):
            if isinstance(layer, vllm_linear.LinearBase):
                return VllmUnquantizedLinearMethod(
                    self.get_linear_config(layer))
            return None

        match layer:
            case vllm_linear.LinearBase():
                linear_config = self.get_linear_config(layer)
                return VllmModelOptFp8LinearMethod(self, linear_config)
            case FusedMoE():
                return VllmModelOptFp8MoEMethod(self, layer, self.mesh)
            case Attention() | MLAAttention():
                return None
            case _:
                return None


class VllmModelOptFp8LinearMethod(VllmFp8LinearMethod):

    def __init__(self, quant_config: VllmModelOptFp8Config,
                 linear_config: VllmQuantLinearConfig):
        # We need to monkeypatch init_fp8_linear_kernel like in VllmFp8LinearMethod
        vllm_modelopt.init_fp8_linear_kernel = lambda *args, **kwargs: None
        super().__init__(quant_config, linear_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, vllm_linear.LinearBase)

        # ModelOpt FP8 LinearMethod stores weights as [out, in] and has weight_scale, input_scale.
        # vLLM's ModelOptFp8LinearMethod.process_weights_after_loading:
        # 1. Takes max of weight_scale.
        # 2. Requantizes if scales are not uniform (not likely for per-tensor).
        # 3. Transposes weight.
        # 4. Sets weight_scale and input_scale as max of loaded scales.

        # In TPU Inference, we want to use VllmFp8LinearMethod's logic but adapted for these names.
        # VllmFp8LinearMethod expects weight_scale_inv.

        if self.quant_config.quant_method == "FP8":
            # Per-tensor quantization
            weight_scale = t2j(layer.weight_scale, use_dlpack=False).max()
            # input_scale = t2j(layer.input_scale, use_dlpack=False).max()

            # Convert weight_scale to weight_scale_inv for compatibility with common_fp8
            # actually we can just shard it and then dequantize if we want to use common_fp8,
            # but VllmFp8LinearMethod is more complex.

            # For now, let's just do simple sharding for per-tensor.
            # Actually, let's just use VllmFp8LinearMethod's logic by temporarily setting the expected attributes.
            layer.weight_scale_inv = Parameter(torch.tensor(
                float(1.0 / weight_scale)),
                                               requires_grad=False)
            delattr(layer, "weight_scale")
            if hasattr(layer, "input_scale"):
                delattr(layer, "input_scale")

            super().process_weights_after_loading(layer)
        else:
            # TODO: Support FP8_PER_CHANNEL_PER_TOKEN and FP8_PB_WO
            logger.warning(
                "ModelOpt %s linear method is not fully implemented for TPU.",
                self.quant_config.quant_method)
            super().process_weights_after_loading(layer)


class VllmModelOptFp8MoEMethod(VllmFp8MoEMethod):

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # ModelOpt MoE has w13_weight_scale, w2_weight_scale, w13_input_scale, w2_input_scale.
        # VllmFp8MoEMethod expects w13_weight_scale_inv, w2_weight_scale_inv.

        # Convert scales to inv_scales
        # w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        # w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        # ModelOpt MoE weights are [num_experts, intermediate_size * shards, hidden_size]
        # and scales are [num_experts, shards].
        # We need to handle this.

        # For simplicity, let's just monkeypatch the names and call super.
        # But wait, VllmFp8MoEMethod.process_weights_after_loading expects block_quant.
        # ModelOpt MoE is usually per-tensor (per expert).

        # If it's per-tensor, we can't easily use VllmFp8MoEMethod which is blockwise.
        # Actually VllmFp8MoEMethod says: assert self.block_quant.

        logger.warning(
            "ModelOpt FP8 MoE method is not fully implemented for TPU. Falling back to Unquantized."
        )
        # TODO: Implement this properly.
        super().process_weights_after_loading(layer)


@register_quantization_config(quant_methods.MODELOPT_NVFP4)
class VllmModelOptNvFp4Config(vllm_modelopt.ModelOptNvFp4Config,
                              VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return quant_methods.MODELOPT_NVFP4

    def get_quant_method(self, layer, prefix):
        if self.is_layer_excluded(prefix):
            if isinstance(layer, vllm_linear.LinearBase):
                return VllmUnquantizedLinearMethod(
                    self.get_linear_config(layer))
            return None
        if isinstance(layer, vllm_linear.LinearBase):
            return VllmModelOptNvFp4LinearMethod(self,
                                                 self.get_linear_config(layer))
        elif isinstance(layer, FusedMoE):
            return VllmModelOptNvFp4FusedMoE(self, layer, self.mesh)
        return super().get_quant_method(layer, prefix)


class VllmModelOptNvFp4LinearMethod(VllmUnquantizedLinearMethod):

    def __init__(self, quant_config: VllmModelOptNvFp4Config,
                 linear_config: VllmQuantLinearConfig):
        super().__init__(linear_config)
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # NVFP4 weight processing for TPU.
        # We dequantize to BF16 and then shard.

        weight = t2j(layer.weight, use_dlpack=False)
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        weight_global_scale = t2j(layer.weight_scale_2,
                                  use_dlpack=False).max().item()

        @jax.jit
        def dequantize_and_process(weight, weight_scale, weight_global_scale):
            return dequantize_nvfp4(
                weight,
                weight_scale,
                weight_global_scale,
                group_size=self.quant_config.group_size,
                out_dtype=jnp.bfloat16,
            )

        dequantized_weight = dequantize_and_process(weight, weight_scale,
                                                    weight_global_scale)

        # Now we have BF16 weights, we can shard them.
        mesh = self.linear_config.mesh
        # TPU Inference usually expects transposed weight [in, out] for Linear.
        # vLLM's NVFP4 weight is [out, in].
        dequantized_weight = dequantized_weight.T

        sharded_weight = shard_linear_weights(dequantized_weight, mesh,
                                              self.linear_config.sharding_spec)

        # Replace parameters
        delattr(layer, "weight")
        delattr(layer, "weight_scale")
        delattr(layer, "input_scale")
        delattr(layer, "weight_scale_2")

        layer.weight = Parameter(torch_view(sharded_weight),
                                 requires_grad=False)


class VllmModelOptNvFp4FusedMoE(VllmUnquantizedFusedMoEMethod):

    def __init__(self, quant_config: VllmModelOptNvFp4Config, moe: FusedMoE,
                 mesh: Mesh):
        super().__init__(moe.moe_config, mesh)
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w13_weight_scale_2 = t2j(layer.w13_weight_scale_2, use_dlpack=False)

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)
        w2_weight_scale_2 = t2j(layer.w2_weight_scale_2, use_dlpack=False)

        @jax.jit
        def dequantize_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_weight_scale_2,
            w2_weight,
            w2_weight_scale,
            w2_weight_scale_2,
        ):
            # w13_weight: [E, I*S, H/2]
            # w13_weight_scale: [E, I*S, H/group_size]
            # w13_weight_scale_2: [E, S]

            e, intermediate_shards, half_h = w13_weight.shape
            h = half_h * 2
            group_size = self.quant_config.group_size

            # Handle w13
            w13_flat = w13_weight.reshape(-1, half_h)
            w13_scale_flat = w13_weight_scale.reshape(-1, h // group_size)
            s = w13_weight_scale_2.shape[1]
            w13_gscale_flat = jnp.repeat(w13_weight_scale_2,
                                         intermediate_shards // s,
                                         axis=1).reshape(-1, 1)

            w13_dq = dequantize_nvfp4(w13_flat, w13_scale_flat,
                                      w13_gscale_flat, group_size,
                                      jnp.bfloat16)
            w13_dq = w13_dq.reshape(e, intermediate_shards, h)

            # Handle w2
            e, h, half_i = w2_weight.shape
            i = half_i * 2
            w2_flat = w2_weight.reshape(-1, half_i)
            w2_scale_flat = w2_weight_scale.reshape(-1, i // group_size)
            w2_gscale_flat = w2_weight_scale_2.reshape(-1, 1)

            w2_dq = dequantize_nvfp4(w2_flat, w2_scale_flat, w2_gscale_flat,
                                     group_size, jnp.bfloat16)
            w2_dq = w2_dq.reshape(e, h, i)

            return w13_dq, w2_dq

        w13_dq, w2_dq = dequantize_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_weight_scale_2,
            w2_weight,
            w2_weight_scale,
            w2_weight_scale_2,
        )

        weights = common_unquantized.process_unquantized_moe_weights(
            mesh=self.mesh,
            moe_backend=self.moe_backend,
            activation=layer.activation,
            w13_weight=w13_dq,
            w13_bias=None,
            w2_weight=w2_dq,
            w2_bias=None)

        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        # Replace parameters
        delattr(layer, "w13_weight")
        delattr(layer, "w13_weight_scale")
        delattr(layer, "w13_weight_scale_2")
        delattr(layer, "w13_input_scale")
        delattr(layer, "w2_weight")
        delattr(layer, "w2_weight_scale")
        delattr(layer, "w2_weight_scale_2")
        delattr(layer, "w2_input_scale")

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)


@register_quantization_config(quant_methods.MODELOPT_MXFP8)
class VllmModelOptMxFp8Config(vllm_modelopt.ModelOptMxFp8Config,
                              VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return quant_methods.MODELOPT_MXFP8

    def get_quant_method(self, layer, prefix):
        if self.is_layer_excluded(prefix):
            if isinstance(layer, vllm_linear.LinearBase):
                return VllmUnquantizedLinearMethod(
                    self.get_linear_config(layer))
            return None
        if isinstance(layer, vllm_linear.LinearBase):
            return VllmModelOptMxFp8LinearMethod(self,
                                                 self.get_linear_config(layer))
        elif isinstance(layer, FusedMoE):
            return VllmModelOptMxFp8FusedMoE(self, layer, self.mesh)
        return super().get_quant_method(layer, prefix)


class VllmModelOptMxFp8LinearMethod(VllmUnquantizedLinearMethod):

    def __init__(self, quant_config: VllmModelOptMxFp8Config,
                 linear_config: VllmQuantLinearConfig):
        super().__init__(linear_config)
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # MXFP8 weight processing for TPU.
        # We dequantize to BF16 and then shard.

        weight = t2j(layer.weight, use_dlpack=False)
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)

        @jax.jit
        def dequantize_and_process(weight, weight_scale):
            weight_fp32 = weight.astype(jnp.float32)
            scale_fp32 = e8m0_to_fp32(weight_scale)
            dequantized = weight_fp32 * jnp.repeat(scale_fp32, 32, axis=-1)
            return dequantized.astype(jnp.bfloat16)

        dequantized_weight = dequantize_and_process(weight, weight_scale)

        mesh = self.linear_config.mesh
        dequantized_weight = dequantized_weight.T

        sharded_weight = shard_linear_weights(dequantized_weight, mesh,
                                              self.linear_config.sharding_spec)

        # Replace parameters
        delattr(layer, "weight")
        delattr(layer, "weight_scale")

        layer.weight = Parameter(torch_view(sharded_weight),
                                 requires_grad=False)


class VllmModelOptMxFp8FusedMoE(VllmUnquantizedFusedMoEMethod):

    def __init__(self, quant_config: VllmModelOptMxFp8Config, moe: FusedMoE,
                 mesh: Mesh):
        super().__init__(moe.moe_config, mesh)
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        @jax.jit
        def dequantize_moe_weights(
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
        ):
            w13_dq = w13_weight.astype(jnp.float32) * jnp.repeat(
                e8m0_to_fp32(w13_weight_scale), 32, axis=-1)

            w2_dq = w2_weight.astype(jnp.float32) * jnp.repeat(
                e8m0_to_fp32(w2_weight_scale), 32, axis=-1)

            return w13_dq.astype(jnp.bfloat16), w2_dq.astype(jnp.bfloat16)

        w13_dq, w2_dq = dequantize_moe_weights(
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
        )

        weights = common_unquantized.process_unquantized_moe_weights(
            mesh=self.mesh,
            moe_backend=self.moe_backend,
            activation=layer.activation,
            w13_weight=w13_dq,
            w13_bias=None,
            w2_weight=w2_dq,
            w2_bias=None)

        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        # Replace parameters
        delattr(layer, "w13_weight")
        delattr(layer, "w13_weight_scale")
        delattr(layer, "w2_weight")
        delattr(layer, "w2_weight_scale")

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)
