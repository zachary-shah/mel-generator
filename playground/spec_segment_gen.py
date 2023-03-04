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

def segment_audio(audio_data, fs=22050, num_segments=5, pitch_augment=True):
    # Split audio into 5 second clips (TODO: remove length hardcoding, make CL arg)
    num_samples_per_segment = num_segments * fs 
    num_segments = int(np.ceil(len(audio_data) / num_samples_per_segment))
    audio_segments = librosa.util.frame(audio_data, frame_length=num_samples_per_segment, hop_length=num_samples_per_segment).T

    # modulate through 12 keys
    if pitch_augment:
        pitch_augmented_segs = []
        # pitch modulation for data augmentation
        for pitch_offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
            for audio_segment in audio_segments:
                pitch_augmented_segs.append(librosa.effects.pitch_shift(audio_segment, sr=fs, n_steps=pitch_offset))
        audio_segments = np.vstack([audio_segments, np.stack(pitch_augmented_segs)])

    # TODO: consider time dilation with librosa.effects.time_stretch(audio_segment, rate=2.0)

    return audio_segments


def write_wav_file(audio_data, file_path, fs, verbose=False):
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
    
    if verbose:
        print(f"Saved audio file at {file_path} with sampling rate {fs} and length {len(audio_data) / fs:.2f} seconds.")

## @neelesh: I added this function to your code to make entire batch generation process collable / removed hardcoding of audio to load in 
'''
generates spectrograms of all audio files 
Parameters
- audio_files: list of strings, where each string is an audio file name (e.g.: ['pop.00000.wav', 'pop.00001.wav'])
- audio_files_dir: root location of audio files in directory (e.g., for case below: '../pop-data')
- output_dir: location of where all files will be saved
- fs: sampling rate, using default = 44100
- verbose: True will turn on print statements 

Saved files located at: 
<outputdir>
    <segments> - folder with all .wav segments
        <audio_file_1>_<segment_1>.wav
        <audio_file_1>_<segment_2>.wav
        ...
        <audio_file_n>_<segment_k>.wav
    <target> - folder with all .png spectrograms saved
        <audio_file_1>_<segment_1>.png
        <audio_file_1>_<segment_2>.png
        ...
        <audio_file_n>_<segment_k>.png
'''
def generate_specs_batch(audio_files, audio_files_dir, output_dir, fs=22050, verbose=False):
    segments_dir = os.path.join(output_dir,"segments")
    os.makedirs(segments_dir, exist_ok=True)

    for audio_file in audio_files:
        audio_filename = audio_file[:audio_file.index(".wav")]

        audio_segments = read_wav_file(os.path.join(audio_files_dir, audio_file), fs=fs)

        pitch_augmented_segs = []
        # pitch modulation for data augmentation
        for pitch_offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
            for audio_segment in audio_segments:
                pitch_augmented_segs.append(librosa.effects.pitch_shift(audio_segment, sr=fs, n_steps=pitch_offset))

        audio_segments = np.vstack([audio_segments, np.stack(pitch_augmented_segs)])

        # TODO: consider time dilation with librosa.effects.time_stretch(audio_segment, rate=2.0)

        for i, segment in enumerate(audio_segments):
            write_wav_file(segment, os.path.join(segments_dir, f'{audio_filename}_seg{i}.wav'), fs=fs,  verbose=verbose)

    # Function is from riffusion/cli.py
    audio_to_images_batch(audio_dir=segments_dir, output_dir=os.path.join(output_dir, "target"))

    if verbose:
        print("Segmentation and spectrogram generation complete.")

## REPLACE CODE BLOCK WITH ONE LINE BELOW:
    # remove hardcoding of the specific audio file to read in
    # audio_segments = read_wav_file('../pop-data/pop.00000.wav', fs=default_fs)

    # make the output directory as subfolders, based on the input audio names
    # pop_audio_dir = "./pop_audio_segments"

    # os.makedirs(pop_audio_dir, exist_ok=True)
    # for i, segment in enumerate(audio_segments):
    #     write_wav_file(segment, f'{pop_audio_dir}/segment_{i}.wav', fs=default_fs)

    # # Log the intermediate audio files as sanity check
    # # Function is from riffusion/cli.py
    # audio_to_images_batch(audio_dir=pop_audio_dir, output_dir='./output_dir/')
# line commented to prevent running every time import
    # generate_specs_batch(['pop.00000.wav'], "../pop-data/", "test_dataset")


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
