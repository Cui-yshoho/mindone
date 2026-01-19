# Copyright 2025 The HuggingFace Team.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import Gemma2Config

import mindspore as ms

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": 1, "dtype": "float16"},
    {"mode": 1, "dtype": "bfloat16"},
    {"mode": 0, "dtype": "float16"},
    {"mode": 0, "dtype": "bfloat16"},
]


@ddt
class SanaVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan",
            dict(
                base_dim=3,
                z_dim=16,
                dim_mult=[1, 1, 1, 1],
                num_res_blocks=1,
                temperal_downsample=[False, True, True],
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
            "mindone.diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
            dict(),
        ],
        [
            "text_encoder",
            "transformers.models.gemma2.modeling_gemma2.Gemma2Model",
            "mindone.transformers.models.gemma2.modeling_gemma2.Gemma2Model",
            dict(
                config=Gemma2Config(
                    head_dim=16,
                    hidden_size=8,
                    initializer_range=0.02,
                    intermediate_size=64,
                    max_position_embeddings=8192,
                    model_type="gemma2",
                    num_attention_heads=2,
                    num_hidden_layers=1,
                    num_key_value_heads=2,
                    vocab_size=8,
                    attn_implementation="eager",
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.gemma.tokenization_gemma.GemmaTokenizer",
            "transformers.models.gemma.tokenization_gemma.GemmaTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/dummy-gemma",
            ),
        ],
        [
            "transformer",
            "diffusers.models.transformers.transformer_sana_video.SanaVideoTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_sana_video.SanaVideoTransformer3DModel",
            dict(
                in_channels=16,
                out_channels=16,
                num_attention_heads=2,
                attention_head_dim=12,
                num_layers=2,
                num_cross_attention_heads=2,
                cross_attention_head_dim=12,
                cross_attention_dim=24,
                caption_channels=8,
                mlp_ratio=2.5,
                dropout=0.0,
                attention_bias=False,
                sample_size=8,
                patch_size=(1, 2, 2),
                norm_elementwise_affine=False,
                norm_eps=1e-6,
                qk_norm="rms_norm_across_heads",
                rope_max_seq_len=32,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "transformer",
                "vae",
                "scheduler",
                "text_encoder",
                "tokenizer",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
            "complex_human_instruction": [],
            "use_resolution_binning": False,
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.sana_video.pipeline_sana_video.SanaVideoPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.sana_video.pipeline_sana_video.SanaVideoPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        if mode == 0:
            ms_pipe.transformer.construct = ms.jit(ms_pipe.transformer.construct)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_video = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_video = ms_pipe(**inputs)

        pt_video_slice = pt_video.frames[0][0, -3:, -3:, -1]
        ms_video_slice = ms_video[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_video_slice - ms_video_slice) / np.linalg.norm(pt_video_slice) < threshold


@ddt
class SanaVideoPipelineIntegrationTests(unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        # TODO: Implement integration test with pretrained model when available
        # For now, skip this test
        self.skipTest("Integration test needs to be implemented with pretrained model")
