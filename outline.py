## OUTLINING REQUIREMENTS FOR PROJECt
import pydub
import numpy as np

### AUDIO PREPROCESSING PIPELINE
# most code can be copied from cli.py (audio_to_images_batch, sample_clips_batch)
# only novelties we need to add are midi compression and splitting audio with paired trimmed / background audio returned

# step 1: convert to midi (optional, but will help with simplification of training for first round)
def midi_compression(audio):
    # basic midi compression pipeline:
        # convert .wav file to .MIDI
        # convert .MIDI back to .wav
    return audio

# step 2: split melody from audio, and return melody-free and melody-containing reference pairs
def split_audio(audio):
    # highlight all locations where a melody exists
    # trim samples to remove areas not containing melody (like beginning or end of a song, for example)
    trimmed_audio = []
    # split each trimmed audio segment and remove melody features to isolate background audio
    # for splitting: reference audio_splitter.py in riffusion github
    background_audio = []

    # add text labels to describe what is present in the audio for text embedding process
    text_labels = []

    return trimmed_audio, background_audio, text_labels

# step 3: clip audio into fixed length segments
def clip_audio(audio, fs, len=5, overlap_percent=0.25):
    # essentially create frames starting at first sample in input_wav of length len * fs
    # start of next segment is shifted by len * fs * overlap_percent
    # return list of all clips
    audio_clips = []
    return audio_clips

# main preprocessing pipeline
def preprocess_audio(input_path, output_path):
    # given input_path which is a folder containing all samples in dataset, preprocess each input sample and save all
    # preprocessed data at output_path

    # get list of files in input_path
    filelist = []

    for file in filelist: 

        # open audio file
        audio = pydub.AudioSegment.from_file(str(audio_path))

        audio = midi_compression(audio)
        full_audio, background_audio = split_audio(audio)
        
        full_audio_list = clip_audio(full_audio, fs, len=5, overlap_percent=.25)
        background_audio_list = clip_audio(background_audio, fs, len=5, overlap_percent=.25)

        # now save all full audio and background audio files to outputpath with some naming convention

### IMAGE PREPROCESSING PIPELINE

# convert audio to spectrogram --> can just use code from riffusion in spectrogram_converter.py and spectrogram_image_converter.py

# riffusion usage

# we still need some text input unfortunately. so this may require us to manually label all our input data with labels:
    # we could make task as simple as something like "voice", "saxophone", "trumpet", etc


# use init_image as where we pass in our images