"""
Data model for the riffusion API.
"""
from __future__ import annotations

import typing as T
from dataclasses import dataclass
from PIL import Image

@dataclass(frozen=True)
class PromptInput:
    """
    Parameters for one end of interpolation.
    """

    # Text prompt fed into a CLIP model
    prompt: str

    # Random seed for denoising
    seed: int

    # Negative prompt to avoid (optional)
    negative_prompt: T.Optional[str] = None

    # Denoising strength (AKA strength)
    denoising: float = 0.75 

    # Classifier-free guidance strength (AKA scale)
    guidance: float = 7.0


# new input datatype better suited for text + image 2 image generation
@dataclass(frozen=True)
class Img2ImgInput:

    # Text prompt fed into a CLIP model
    text_prompt: str

    # Random seed for denoising
    seed: int

    # path to initial spectrogram as an image
    init_spectrogram: Image.Image

    # mask for spectrogram
    mask_image: T.Optional[Image.Image] = None

    # Negative text prompt to avoid (optional)
    negative_prompt: T.Optional[str] = None

    # Denoising strength (AKA strength)
    denoising: float = 0.75 

    # Classifier-free guidance strength. guidance=1 corresponds to no guidance.
    # guidance > 1 strengthens effect of guidance. guidance is necessary for effective
    # text conditioning accordign to Imagen paper
    guidance: float = 7.0

    # number of diffusion sampling steps
    ddim_steps: int = 50 

    # parameter for diffusion. 0.0 corresponds to deterministic sampling
        # eta (η) is only used with the DDIMScheduler, and should be between [0, 1]
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    ddim_eta: float = 0.0 # number

    # number of latent channels (may not need?)
    C: int = 4

    # downsampling factor
    f: int = 8

    # already taken care of
    # # scale for unconditional guidance
    # scale: float = 5.0
    # # strength for noising / denoising. strength=1 corresponds to full destruction of information from input image
    # strength: float = 0.75

    # other
    # output directory
    # skip saving grid (skip_grid)
    # skip saving individual samples (skip_save)
    # fixed_code: if enabled, uses same starting code across all samples (dont need if doing only 1 sample)
    # n_iter (sample this often)


@dataclass(frozen=True)
class InferenceOutput:
    """
    Response from the model inference server.
    """

    # base64 encoded spectrogram image as a JPEG
    image: str

    # base64 encoded audio clip as an MP3
    audio: str

    # The duration of the audio clip
    duration_s: float



@dataclass(frozen=True)
class InferenceInput:
    """
    Parameters for a single run of the riffusion model, interpolating between
    a start and end set of PromptInputs. This is the API required for a request
    to the model server.
    """

    # Start point of interpolation
    start: PromptInput

    # End point of interpolation
    end: PromptInput

    # Interpolation alpha [0, 1]. A value of 0 uses start fully, a value of 1
    # uses end fully.
    alpha: float

    # Number of inner loops of the diffusion model
    num_inference_steps: int = 50

    # Which seed image to use
    seed_image_id: str = "og_beat"

    # ID of mask image to use
    mask_image_id: T.Optional[str] = None