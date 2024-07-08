# MindONE

This repository contains SoTA algorithms, models, and interesting projects in the area of content generation, including ChatGPT detection and Stable Diffusion, and will be continously updated.

ONE is short for "ONE for all" and "Optimal generators with No Exception" (credits to GPT-4).
## News

- 2024.07.09 [mindone/diffusers](mindone/diffusers) now supports [Kolors](https://huggingface.co/Kwai-Kolors/Kolors).

**Hello MindSpore** from **Stable Diffusion 3**!

<div>
<img src="https://github.com/townwish4git/mindone/assets/143256262/8c25ae9a-67b1-436f-abf6-eca36738cd17" alt="sd3" width="512" height="512">
</div>

- 2024.06.13 🚀🚀🚀 [mindone/diffusers](mindone/diffusers) now supports [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium). Give it a try yourself!

    ```py
    import mindspore
    from mindone.diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        mindspore_dtype=mindspore.float16,
    )
    prompt = "A cat holding a sign that says 'Hello MindSpore'"
    image = pipe(prompt)[0][0]
    image.save("sd3.png")
    ```

- 2024.05.23
    1. Two OpenSora models are supported!
        - [hpcai-OpenSora](examples/opensora_hpcai) based on VAE+STDiT
        - [PKU-OpenSora](examples/opensora_pku) based on CausalVAE3D+Latte_T2V
    2. [diffusers](mindone/diffusers) is now runnable with MindSpore (experimental)
- 2024.03.22
    1. New diffusion transformer models released!
        - [DiT](examples/dit) for image generation
        - [Latte](examples/latte) for video generation
- 2024.03.04
    1. New generative models released!
        - [AnimateDiff](examples/animatediff) v1, v2, and v3
        - [Pangu Draw v3](examples/pangu_draw_v3) for Chinese text-to-image generation
        - [Stable Video Diffusion(SVD)](examples/svd) for image-to-video generation
        - [Tune-a-Video](examples/tuneavideo) for one-shot video tuning.
    2. Enhanced Stable Diffusion and Stable Diffusion XL with more add-ons: ControlNet, T2I-Adapter, and IP-Adapter.
- 2023.07.01 stable diffusion 2.0 lora fine-tune example can be found [here](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_v2/lora_finetune.md)

## Playground

- [ChatGPT Detection](examples/detect_chatgpt): Detect whether the input texts are generated by ChatGPT

- [Stable Diffusion 1.5/2.x](examples/stable_diffusion_v2): Text-to-image generation via latent diffusion models (with support for inference and finetuning)

- [Stable Diffusion XL](examples/stable_diffusion_xl): New state-of-the-art SD model with double text embedders and larger UNet.

- [VideoComposer](examples/videocomposer): Generate videos with prompts or reference videos via controllable video diffusion (both training and inference are supported)

- [AnimateDiff](examples/animatediff): SoTA text-to-video generation models (including v1, v2, and v3) supporting motion lora fine-tuning.


## Awesome List

- [Awesome Large Vision and Foundation Models](awesome_vision.md)
