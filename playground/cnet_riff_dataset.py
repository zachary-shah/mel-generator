import json
import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path

# for spectrum generation
from spec_segment_gen import generate_specs_batch

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
def create_prompt_file(rootdir="", manual_prompts=None, verbose=False):
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
        
    create_prompt_file(rootdir=output_dir, manual_prompts=manual_prompts, verbose=verbose)

    return