import wave
import numpy as np
import librosa

import os
import sys

sys.path.append('../riffusion-reference/riffusion/')
sys.path.append('../riffusion-reference/')

default_fs = 44100

# TODO: VScode complains, resolve imports properly
from riffusion.util import audio_util, torch_util
from riffusion.cli import audio_to_images_batch 
from riffusion.audio_splitter import split_audio 

import pydub

# Notes on terminology:
# - clip: 5 second (or some fixed length) audio
# - stem: a specific musical component of a given clip
# - segment: synonymous with stem

def read_wav_file(file_path, fs):
    audio_data, sr = librosa.load(file_path, sr=fs, mono=True)

    # Split audio into 5 second clips (TODO: remove length hardcoding, make CL arg)
    num_samples_per_segment = 5 * fs 
    num_segments = int(np.ceil(len(audio_data) / num_samples_per_segment))
    audio_segments = librosa.util.frame(audio_data, frame_length=num_samples_per_segment, hop_length=num_samples_per_segment).T

    return audio_segments

def write_wav_file(audio_data, file_path, fs):
    with wave.open(file_path, 'w') as wave_file:
        num_channels = 1 # Mono audio
        sample_width = 2 # 16-bit audio
        num_frames = len(audio_data)
        comp_type = 'NONE'
        comp_name = 'not compressed'
        wave_params = (num_channels, sample_width, fs, num_frames, comp_type, comp_name)
        # Wave library expects a pretty specific set of info
        wave_file.setparams(wave_params)

        # Scale the audio data to fit within the range of a 16-bit integer
        # Without this, output sounds like garbage
        audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = np.int16(audio_data * 2**15-1)

        # Write the audio data to the file
        wave_file.writeframes(audio_data.tobytes())

    print(f'Saved audio file at {file_path} with sampling rate {fs} and length {len(audio_data) / fs:.2f} seconds.')

# TODO: remove hardcoding of the specific audio file to read in
audio_segments = read_wav_file('../pop-data/pop.00000.wav', fs=default_fs)

# TODO: make the output directory as subfolders, based on the input audio names
pop_audio_dir = "./pop_audio_segments"

os.makedirs(pop_audio_dir, exist_ok=True)
for i, segment in enumerate(audio_segments):
    write_wav_file(segment, f'{pop_audio_dir}/segment_{i}.wav', fs=default_fs)

# Log the intermediate audio files as sanity check
# Function is from riffusion/cli.py
audio_to_images_batch(audio_dir=pop_audio_dir, output_dir='./output_dir/')

# In progress: Operating demucs on the audio segments to further divide the 5-second clips into stems
# First, use the first 5 second clip of the very first pop audio as an example

# Commented below out for now

# segment_idx_list = [0]
# for i in segment_idx_list:
#     print(i)
#     segment_file_name = f'{pop_audio_dir}/segment_{i}.wav'
#     print(segment_file_name)
#     inp_audio_clip = pydub.AudioSegment.from_file(segment_file_name, format="wav")
#     audio_stems = split_audio(segment=inp_audio_clip) # violates naming convention but oh well for now
#     stems_dir = f'{pop_audio_dir}/segment_{i}'
#     os.makedirs(pop_audio_dir, exist_ok=True)
#     for j, audio_stem in enumerate(audio_stems):
#         write_wav_file(audio_stem, f'{stems_dir}/segment_{i}/stem_{j}.wav', fs=default_fs)
