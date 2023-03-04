import json
import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path

# for spectrum generation
from spec_segment_gen import generate_specs_batch, segment_audio, write_wav_file
from external import riffusion_tools
import spleeter_utils

# training dataset built off reference data with the following structure
# <rootdir>: fullfile path that contains:
    # prompt.json --> list of json files in the form {"source": "imgpath", "target": "targetpath", "prompt":, "prompt-str"}
    # source --> folder with canny edge detection spectrograms
    # target --> folder with full audio spectrograms
class CnetRiffDataset(Dataset):
    def __init__(self, rootdir):
        self.data = []
        self.rootdir = rootdir
        with open(os.path.join(rootdir, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.rootdir, source_filename))
        target = cv2.imread(os.path.join(self.rootdir, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

# generate source image with same name as target_name in target folder and save it in source folder
# target must be saved at rootdir/target/target_filename
# source will be saved to rootdir/source/target_filename
# TODO: figure out low/high thresh for canny edge detection
def generate_canny_source(target_filename, rootdir="", low_thres=100, high_thres=200):

    # check threshold values 
    assert low_thres > 1 and low_thres<255, f"Threshold out of bounds; must be between 1 and 255"
    assert low_thres > 1 and high_thres<255, f"Threshold out of bounds; must be between 1 and 255"
    
    # open image
    target = cv2.imread(os.path.join(rootdir, "target", target_filename))
    
    # may not need this line? 
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    # run canny edge detection
    source = cv2.Canny(target, low_thres, high_thres)
    
    # save at proper place
    if not os.path.exists(os.path.join(rootdir, "source")):
        os.makedirs(os.path.join(rootdir, "source"))
    cv2.imwrite(os.path.join(rootdir, "source", target_filename), source)
    return

## TODO: generate intellegent prompts
def generate_target_prompt(target_filename, rootdir="", manual_prompt=None):    
    if manual_prompt is None:
        # TODO: intellegent prompt generation from target
        #target = cv2.imread(os.path.join(rootdir, "target", target_filename))
        prompt = "dummy prompt"
        pass
    else:
        prompt = manual_prompt
    return prompt

# it is assumed all sources and targets images created at this point and saved
# either create off manual list of prompts or generate prompt intelligently
def create_prompt_file_old(rootdir="", manual_prompts=None, verbose=False):
    # remove prompts.json if it already exists; rewrite it fresh
    if Path(os.path.join(rootdir,"prompt.json")).is_file():
        os.remove(os.path.join(rootdir,"prompt.json"))

    i = 0
    with open(os.path.join(rootdir,"prompt.json"), 'w') as outfile:
        # clear json file to prevent double writing
        for file in os.listdir(os.path.join(rootdir, "source")):
            # check the files which are end with specific extension
            if file.endswith(".png") or file.endswith(".jpg"):
                # print path name of selected files
                if manual_prompts is None:
                    prompt = generate_target_prompt(file, rootdir=rootdir)
                else:
                    prompt = manual_prompts[i]
                    i += 1

                packet = {
                    "source": os.path.join("source", file),
                    "target": os.path.join("target", file),
                    "prompt": prompt
                }
                json.dump(packet, outfile)
                outfile.write('\n')
    if verbose:
        print(f"Successfully generated prompts for {i} targets")
    return

# wrapper for all training data preprocessing
def generate_training_data_from_audio(audio_files, audio_files_dir, output_dir, manual_prompts=None, verbose=False):

    # do segmentation and generate spectrograms 
    generate_specs_batch(audio_files, audio_files_dir, output_dir, verbose=verbose)

    # get filenames 
    for file in os.listdir(os.path.join(output_dir, "target")):
        generate_canny_source(file, rootdir=output_dir)
        
    create_prompt_file_old(rootdir=output_dir, manual_prompts=manual_prompts, verbose=verbose)

    return

# create fresh prompt file
def create_prompt_file(rootdir):
    # remove prompts.json if it already exists; rewrite it fresh
    if Path(os.path.join(rootdir,"prompt.json")).is_file():
        os.remove(os.path.join(rootdir,"prompt.json"))
    return

# append for all segment source/targets created for one audio file
# all sources and files have same prompt for now
def append_to_prompt_file(rootdir, source_filepaths, target_filepaths, prompt, verbose=False):

    with open(os.path.join(rootdir,"prompt.json"), 'w') as outfile:
        for i in range(len(source_filepaths)):
            packet = {
                "source": str(source_filepaths[i]),
                "target": str(target_filepaths[i]),
                "prompt": str(prompt)
            }
            json.dump(packet, outfile)
            outfile.write('\n')
    if verbose:
        print(f"Successfully generated prompts for {i} training examples")
    return

def generate_and_replace_canny_source(file_path, low_thres=100, high_thres=200):

    # check threshold values 
    assert low_thres > 1 and low_thres<255, f"Threshold out of bounds; must be between 1 and 255"
    assert low_thres > 1 and high_thres<255, f"Threshold out of bounds; must be between 1 and 255"
    
    # open image
    accompaniment_spec = cv2.imread(file_path)
    # flip color scheme
    accompaniment_spec = cv2.cvtColor(accompaniment_spec, cv2.COLOR_BGR2RGB)
    # run canny edge detection
    source_spec = cv2.Canny(accompaniment_spec, low_thres, high_thres)
    cv2.imwrite(file_path, source_spec)
    
    return

# given audio files, save all targets, source, and prompt file
def preprocess_batch(audio_files, audio_files_dir, output_dir, fs=22050, verbose=False, save_wav=False):

    create_prompt_file(rootdir=output_dir)
    
    segments_dir = os.path.join(output_dir,"segment")
    os.makedirs(segments_dir, exist_ok=True)

    targets_dir = os.path.join(output_dir,"target")
    os.makedirs(targets_dir, exist_ok=True)

    sources_dir = os.path.join(output_dir,"source")
    os.makedirs(sources_dir, exist_ok=True)

    for audio_file in audio_files:
        audio_filename = audio_file[:audio_file.index(".wav")]

        # audio splitting
        splits = spleeter_utils.separate_audio(os.path.join(audio_files_dir, audio_file), fs=fs, stem_num=2)
        accompaniment_audio = splits['accompaniment']
        full_audio = splits['full_audio']

        # get audio segments with pitch augmentation on (should be 72 segments total)
        full_audio_segments = segment_audio(full_audio, fs=fs, num_segments=5, pitch_augment=True)
        accompaniment_audio_segments = segment_audio(accompaniment_audio, fs=fs, num_segments=5, pitch_augment=True)

        # generally, don't save .wav files as this is will require too much storage
        if save_wav:
            for i, segment in enumerate(full_audio_segments):
                write_wav_file(segment, os.path.join(segments_dir, f'{audio_filename}_seg{i}_full.wav'), fs=fs,  verbose=verbose)
            for i, segment in enumerate(accompaniment_audio_segments):
                write_wav_file(segment, os.path.join(segments_dir, f'{audio_filename}_seg{i}_bgnd.wav'), fs=fs,  verbose=verbose)
        
        # make paths for saving targets
        target_save_paths = []
        for i in range(full_audio_segments.shape[0]):
            target_save_paths.append(Path(os.path.join(targets_dir, f'{audio_filename}_seg{i}.wav')))
    
        # save target spectrograms
        riffusion_tools.audio_to_images_batch(segment_arr = full_audio_segments,
                                             output_paths = target_save_paths,
                                             sample_rate = fs)
        
        # save source spectrograms
        source_save_paths = []
        for i in range(accompaniment_audio_segments.shape[0]):
            source_save_paths.append(Path(os.path.join(sources_dir, f'{audio_filename}_seg{i}.wav')))
        riffusion_tools.audio_to_images_batch(segment_arr = accompaniment_audio_segments,
                                             output_paths = source_save_paths,
                                             sample_rate = fs)
        
        # turn sources into canny edges
        for i in range(len(source_save_paths)):
            generate_and_replace_canny_source(source_save_paths[i], low_thres=100, high_thres=200)
        
        # append to prompt file
        append_to_prompt_file(rootdir=output_dir, 
                            source_fileppaths=source_save_paths, 
                            target_filepaths=target_save_paths,
                            prompt="Generate a pop melody.",
                            verbose=verbose)
    if verbose:
        print("Segmentation and spectrogram generation complete.")
    return
