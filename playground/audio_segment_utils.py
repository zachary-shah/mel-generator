import wave
import numpy as np
import librosa

import os
import sys

default_fs = 44100

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
    
    #print(f"segment_audio: shape: {np.shape(audio_data)}")
    audio_segments = librosa.util.frame(audio_data.T, frame_length=num_samples_per_segment, hop_length=num_samples_per_segment).T

    # modulate through 12 keys
    if pitch_augment:
        pitch_augmented_segs = []
        # pitch modulation for data augmentation
        for pitch_offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
            for audio_segment in audio_segments:
                pitch_augmented_segs.append(librosa.effects.pitch_shift(np.squeeze(audio_segment), sr=fs, n_steps=pitch_offset))
        audio_segments = np.vstack([np.squeeze(audio_segments), np.stack(pitch_augmented_segs)])

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