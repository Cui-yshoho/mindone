from typing import TYPE_CHECKING

from ..utils import _LazyModule

# These modules contain pipelines from multiple libraries/frameworks
_import_structure = {
    "consistency_models": ["ConsistencyModelPipeline"],
    "ddim": ["DDIMPipeline"],
    "ddpm": ["DDPMPipeline"],
    "latent_consistency_models": [
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
    ],
    "stable_diffusion": [
        "StableDiffusionPipeline",
        "StableDiffusionImg2ImgPipeline",
    ],
    "stable_diffusion_xl": [
        "StableDiffusionXLPipeline",
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLImg2ImgPipeline",
    ],
    "pipeline_utils": [
        "DiffusionPipeline",
        "ImagePipelineOutput",
    ],
    "stable_cascade": [
        "StableCascadeCombinedPipeline",
        "StableCascadeDecoderPipeline",
        "StableCascadePriorPipeline",
    ],
    "wuerstchen": [
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ],
    "stable_video_diffusion": ["StableVideoDiffusionPipeline"],
    "kandinsky": [
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyImg2ImgPipeline",
        "KandinskyInpaintCombinedPipeline",
        "KandinskyInpaintPipeline",
        "KandinskyPipeline",
        "KandinskyPriorPipeline",
    ]
}

if TYPE_CHECKING:
    from .consistency_models import ConsistencyModelPipeline
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .latent_consistency_models import LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline
    from .pipeline_utils import DiffusionPipeline, ImagePipelineOutput
    from .stable_cascade import StableCascadeCombinedPipeline, StableCascadeDecoderPipeline, StableCascadePriorPipeline
    from .stable_diffusion import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
    from .stable_diffusion_xl import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )
    from .wuerstchen import (
        WuerstchenCombinedPipeline,
        WuerstchenDecoderPipeline,
        WuerstchenPriorPipeline,
    )
    from .stable_video_diffusion import StableVideoDiffusionPipeline
    from .kandinsky import (
        KandinskyCombinedPipeline,
        KandinskyImg2ImgCombinedPipeline,
        KandinskyImg2ImgPipeline,
        KandinskyInpaintCombinedPipeline,
        KandinskyInpaintPipeline,
        KandinskyPipeline,
        KandinskyPriorPipeline,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
