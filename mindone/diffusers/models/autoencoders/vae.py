# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from ...utils import BaseOutput
from ..activations import SiLU
from ..normalization import GroupNorm
from ..unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: ms.Tensor


class Encoder(nn.Cell):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

        # down
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            down_blocks.append(down_block)
        self.down_blocks = nn.CellList(down_blocks)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], conv_out_channels, 3, pad_mode="pad", padding=1, has_bias=True
        )

        self._gradient_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value=False):
        self._gradient_checkpointing = value
        for down_block in self.down_blocks:
            down_block._recompute(value)
        self.mid_block._recompute(value)

    def construct(self, sample: ms.Tensor) -> ms.Tensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Cell):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = nn.CellList(up_blocks)

        # out
        if norm_type == "spatial":
            raise NotImplementedError("SpatialNorm is not implemented.")
        else:
            self.conv_norm_out = GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, pad_mode="pad", padding=1, has_bias=True)

        self._gradient_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value=False):
        self._gradient_checkpointing = value
        self.mid_block._recompute(value)
        for up_block in self.up_blocks:
            up_block._recompute(value)

    def construct(
        self,
        sample: ms.Tensor,
        latent_embeds: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample, latent_embeds)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class VectorQuantizer(nn.Cell):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e: int,
        vq_embed_dim: int,
        beta: float,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.embedding_table.set_dtype(
            ops.uniform(self.embedding.embedding_table.shape, ms.Tensor(-1.0 / self.n_e), ms.Tensor(1.0 / self.n_e))
        )

        self.remap = remap
        if self.remap is not None:
            self.used = ms.Tensor(np.load(self.remap))
            self.used: ms.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: ms.Tensor) -> ms.Tensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds.dtype)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = ops.randint(0, self.re_embed, size=new[unknown].shape)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: ms.Tensor) -> ms.Tensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds.dtype)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = ops.gather_elements(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def construct(self, z: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor, Tuple]:
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = ops.argmin(ops.cdist(z_flattened, self.embedding.weight), axis=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * ops.mean((ops.stop_gradient(z_q) - z) ** 2) + ops.mean((z_q - ops.stop_gradient(z)) ** 2)
        else:
            loss = ops.mean((ops.stop_gradient(z_q) - z) ** 2) + self.beta * ops.mean((z_q - ops.stop_gradient(z)) ** 2)

        # preserve gradients
        z_q: ms.Tensor = z + ops.stop_gradient(z_q - z)

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2)

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: ms.Tensor, shape: Tuple[int, ...]) -> ms.Tensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q: ms.Tensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2)

        return z_q


@ms.jit_class
class DiagonalGaussianDistribution(object):
    def __init__(self, deterministic: bool = False):
        self.deterministic = deterministic

    def init(self, parameters: ms.Tensor) -> Tuple[ms.Tensor, ...]:
        mean, logvar = ops.chunk(parameters, 2, axis=1)
        logvar = ops.clamp(logvar, -30.0, 20.0)
        if self.deterministic:
            var = std = ops.zeros_like(mean, dtype=parameters.dtype)
        else:
            var = ops.exp(logvar)
            std = ops.exp(0.5 * logvar)
        return mean, logvar, var, std

    def sample(self, parameters: ms.Tensor) -> ms.Tensor:
        mean, logvar, var, std = self.init(parameters)
        # make sure sample is on the same device as the parameters and has same dtype
        sample = ops.randn(
            mean.shape,
            dtype=parameters.dtype,
        )
        x = mean + std * sample
        return x

    def kl(self, parameters: ms.Tensor, other: ms.Tensor = None) -> ms.Tensor:
        mean, logvar, var, std = self.init(parameters)
        if self.deterministic:
            return ms.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * ops.sum(
                    ops.pow(mean, 2) + var - 1.0 - logvar,
                    dim=[1, 2, 3],
                )
            else:
                other_mean, other_logvar, other_var, other_std = self.init(other)
                # fmt: off
                return 0.5 * ops.sum(
                    ops.pow(mean - other_mean, 2) / other_var
                    + var / other_var
                    - 1.0
                    - logvar
                    + other_logvar,
                    dim=[1, 2, 3],
                )
                # fmt: on

    def nll(self, parameters: ms.Tensor, sample: ms.Tensor, dims: Tuple[int, ...] = (1, 2, 3)) -> ms.Tensor:
        mean, logvar, var, std = self.init(parameters)
        if self.deterministic:
            return ms.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * ops.sum(
            logtwopi + logvar + ops.pow(sample - mean, 2) / var,
            dim=dims,
        )

    def mode(self, parameters: ms.Tensor) -> ms.Tensor:
        mean, logvar, var, std = self.init(parameters)
        return mean
