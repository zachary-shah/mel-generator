"""
Command line tools for riffusion.
"""

import random
import typing as T
from multiprocessing.pool import ThreadPool
from pathlib import Path
import sys

import argh
import numpy as np
import pydub
import tqdm
from PIL import Image

sys.path.append('../riffusion-reference/riffusion/')
sys.path.append('../riffusion-reference/')

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams


def audio_to_images_batch(
    audio_segment_arr: np.ndarray,
    audio_paths: list,
    image_extension: str = "jpg",
    step_size_ms: int = 10,
    num_frequencies: int = 512,
    min_frequency: int = 0,
    max_frequency: int = 10000,
    power_for_image: float = 0.25,
    mono: bool = False,
    sample_rate: int = 44100,
    device: str = "cuda",
    num_threads: T.Optional[int] = None,
    limit: int = -1,
):
    """
    Process audio clips into spectrograms in batch, multi-threaded.
    """

    inds = np.arange(len(audio_paths))

    if limit > 0:
        audio_paths = audio_paths[:limit]

    params = SpectrogramParams(
        step_size_ms=step_size_ms,
        num_frequencies=num_frequencies,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        power_for_image=power_for_image,
        stereo=not mono,
        sample_rate=sample_rate,
    )

    converter = SpectrogramImageConverter(params=params, device=device)

    def process_one(ind) -> None:
        # Load into pydub
        try:
            #segment = pydub.AudioSegment.from_file(str(audio_path))
            # read from numpy audiosegment
            segment = pydub.AudioSegment(
                        audio_segment_arr[ind].tobytes(), 
                        frame_rate= sample_rate,
                        sample_width=audio_segment_arr[ind].dtype.itemsize, 
                        channels=1, #TODO: update according to mono boolean
                    )
        except Exception:
            return

        if mono and segment.channels != 1:
            segment = segment.set_channels(1)
        elif not mono and segment.channels != 2:
            segment = segment.set_channels(2)

        # Frame rate
        if segment.frame_rate != params.sample_rate:
            segment = segment.set_frame_rate(params.sample_rate)

        # Convert
        image = converter.spectrogram_image_from_audio(segment)

        # Save
        # print(image_extension)
        image_path = audio_paths[ind].with_suffix("."+image_extension)
        image_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}[image_extension]
        image.save(image_path, exif=image.getexif(), format=image_format)

    # Create thread pool
    pool = ThreadPool(processes=num_threads)
    with tqdm.tqdm(total=len(audio_paths)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(process_one, inds)):
            pbar.update()
