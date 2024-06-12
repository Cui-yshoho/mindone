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

from typing import Callable, Dict, List, Optional, Union

import numpy as np
from transformers import CLIPTokenizer

import mindspore as ms
from mindspore import ops

from mindone.transformers import CLIPTextModel

from ...models import StableCascadeUNet
from ...schedulers import DDPMWuerstchenScheduler
from ...utils import logging, replace_example_docstring
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ..wuerstchen.modeling_paella_vq_model import PaellaVQModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import mindspore as ms
        >>> from mindone.diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline

        >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(
        ...     "stabilityai/stable-cascade-prior", mindspore_dtype=ms.float16
        ... )
        >>> gen_pipe = StableCascadeDecoderPipeline.from_pretrain(
        ...     "stabilityai/stable-cascade", mindspore_dtype=ms.float16
        ... )

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> prior_output = pipe(prompt)
        >>> images = gen_pipe(prior_output[0], prompt=prompt)
        ```
"""


class StableCascadeDecoderPipeline(DiffusionPipeline):
    """
    Pipeline for generating images from the Stable Cascade model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer (`CLIPTokenizer`):
            The CLIP tokenizer.
        text_encoder (`CLIPTextModel`):
            The CLIP text encoder.
        decoder ([`StableCascadeUNet`]):
            The Stable Cascade decoder unet.
        vqgan ([`PaellaVQModel`]):
            The VQGAN model.
        scheduler ([`DDPMWuerstchenScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        latent_dim_scale (float, `optional`, defaults to 10.67):
            Multiplier to determine the VQ latent space size from the image embeddings. If the image embeddings are
            height=24 and width=24, the VQ latent shape needs to be height=int(24*10.67)=256 and
            width=int(24*10.67)=256 in order to match the training conditions.
    """

    unet_name = "decoder"
    text_encoder_name = "text_encoder"
    model_cpu_offload_seq = "text_encoder->decoder->vqgan"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds_pooled",
        "negative_prompt_embeds",
        "image_embeddings",
    ]

    def __init__(
        self,
        decoder: StableCascadeUNet,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        scheduler: DDPMWuerstchenScheduler,
        vqgan: PaellaVQModel,
        latent_dim_scale: float = 10.67,
    ) -> None:
        super().__init__()
        self.register_modules(
            decoder=decoder,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            vqgan=vqgan,
        )
        self.register_to_config(latent_dim_scale=latent_dim_scale)

    def prepare_latents(
        self, batch_size, image_embeddings, num_images_per_prompt, dtype, generator, latents, scheduler
    ):
        _, channels, height, width = image_embeddings.shape
        latents_shape = (
            batch_size * num_images_per_prompt,
            4,
            int(height * self.config.latent_dim_scale),
            int(width * self.config.latent_dim_scale),
        )

        if latents is None:
            latents = randn_tensor(latents_shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        latents = latents * scheduler.init_noise_sigma
        return latents

    def encode_prompt(
        self,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt=None,
        negative_prompt=None,
        prompt_embeds: Optional[ms.Tensor] = None,
        prompt_embeds_pooled: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds_pooled: Optional[ms.Tensor] = None,
    ):
        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = ms.Tensor(text_inputs.input_ids)
            attention_mask = ms.Tensor(text_inputs.attention_mask)

            untruncated_ids = ms.Tensor(self.tokenizer(prompt, padding="longest", return_tensors="np").input_ids)

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not ops.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
                attention_mask = attention_mask[:, : self.tokenizer.model_max_length]

            text_encoder_output = self.text_encoder(
                text_input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            prompt_embeds = text_encoder_output[2][-1]
            if prompt_embeds_pooled is None:
                prompt_embeds_pooled = text_encoder_output[0].unsqueeze(1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype)
        prompt_embeds_pooled = prompt_embeds_pooled.to(dtype=self.text_encoder.dtype)
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds_pooled = prompt_embeds_pooled.repeat_interleave(num_images_per_prompt, dim=0)

        if negative_prompt_embeds is None and do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_prompt_embeds_text_encoder_output = self.text_encoder(
                ms.Tensor(uncond_input.input_ids),
                attention_mask=ms.Tensor(uncond_input.attention_mask),
                output_hidden_states=True,
            )

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output[2][-1]
            negative_prompt_embeds_pooled = negative_prompt_embeds_text_encoder_output[0].unsqueeze(1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype)
            negative_prompt_embeds = negative_prompt_embeds.tile((1, num_images_per_prompt, 1))
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            seq_len = negative_prompt_embeds_pooled.shape[1]
            negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.to(dtype=self.text_encoder.dtype)
            negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.tile((1, num_images_per_prompt, 1))
            negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            # done duplicates

        return prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds, negative_prompt_embeds_pooled

    def check_inputs(
        self,
        prompt,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image_embeddings: Union[ms.Tensor, List[ms.Tensor]],
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        prompt_embeds_pooled: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds_pooled: Optional[ms.Tensor] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image_embedding (`ms.Tensor` or `List[ms.Tensor]`):
                Image Embeddings either extracted from an image or generated by a Prior Model.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            num_inference_steps (`int`, *optional*, defaults to 12):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 0.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `decoder_guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting
                `decoder_guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely
                linked to the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `decoder_guidance_scale` is less than `1`).
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_embeds_pooled (`ms.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_embeds_pooled (`ms.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds_pooled will be generated from `negative_prompt` input
                argument.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [np.random.Generator(s)](https://numpy.org/doc/stable/reference/random/generator.html)
                to make generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"ms"` (`ms.Tensor`).
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple` [`~pipelines.ImagePipelineOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is a list with the generated image
            embeddings.
        """

        # 0. Define commonly used variables
        dtype = next(self.decoder.get_parameters()).dtype
        self._guidance_scale = guidance_scale

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        if isinstance(image_embeddings, list):
            image_embeddings = ops.cat(image_embeddings, axis=0)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Compute the effective number of images per prompt
        # We must account for the fact that the image embeddings from the prior can be generated with num_images_per_prompt > 1
        # This results in a case where a single prompt is associated with multiple image embeddings
        # Divide the number of image embeddings by the batch size to determine if this is the case.
        num_images_per_prompt = num_images_per_prompt * (image_embeddings.shape[0] // batch_size)

        # 2. Encode caption
        if prompt_embeds is None and negative_prompt_embeds is None:
            _, prompt_embeds_pooled, _, negative_prompt_embeds_pooled = self.encode_prompt(
                prompt=prompt,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_pooled=prompt_embeds_pooled,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
            )

        # The pooled embeds from the prior are pooled again before being passed to the decoder
        prompt_embeds_pooled = (
            ops.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
            if self.do_classifier_free_guidance
            else prompt_embeds_pooled
        )
        effnet = (
            ops.cat([image_embeddings, ops.zeros_like(image_embeddings)])
            if self.do_classifier_free_guidance
            else image_embeddings
        )

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latents
        latents = self.prepare_latents(
            batch_size, image_embeddings, num_images_per_prompt, dtype, generator, latents, self.scheduler
        )

        # 6. Run denoising loop
        self._num_timesteps = len(timesteps[:-1])
        for i, t in enumerate(self.progress_bar(timesteps[:-1])):
            timestep_ratio = t.broadcast_to((latents.shape[0],)).to(dtype)

            # 7. Denoise latents
            predicted_latents = self.decoder(
                sample=ops.cat([latents] * 2) if self.do_classifier_free_guidance else latents,
                timestep_ratio=ops.cat([timestep_ratio] * 2) if self.do_classifier_free_guidance else timestep_ratio,
                clip_text_pooled=prompt_embeds_pooled,
                effnet=effnet,
                return_dict=False,
            )[0]

            # 8. Check for classifier free guidance and apply it
            if self.do_classifier_free_guidance:
                predicted_latents_text, predicted_latents_uncond = predicted_latents.chunk(2)
                predicted_latents = ops.lerp(
                    predicted_latents_uncond,
                    predicted_latents_text,
                    ms.tensor(self.guidance_scale, dtype=predicted_latents_text.dtype),
                )

            # 9. Renoise latents to next timestep
            # TODO: check prev_sample
            latents = self.scheduler.step(
                model_output=predicted_latents,
                timestep=timestep_ratio,
                sample=latents,
                generator=generator,
            )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        if output_type not in ["ms", "np", "pil", "latent"]:
            raise ValueError(
                f"Only the output types `ms`, `np`, `pil` and `latent` are supported not output_type={output_type}"
            )

        if not output_type == "latent":
            # 10. Scale and decode the image latents with vq-vae
            # TODO: check self.vqgan.decode(latents).sample.clamp(0, 1)
            latents = self.vqgan.config.scale_factor * latents
            images = self.vqgan.decode(latents)[0].clamp(0, 1)
            if output_type == "np":
                images = images.permute((0, 2, 3, 1)).float().asnumpy()  # float() as bfloat16-> numpy doesnt work
            elif output_type == "pil":
                images = images.permute((0, 2, 3, 1)).float().asnumpy()  # float() as bfloat16-> numpy doesnt work
                images = self.numpy_to_pil(images)
        else:
            images = latents

        if not return_dict:
            return images
        return ImagePipelineOutput(images)
