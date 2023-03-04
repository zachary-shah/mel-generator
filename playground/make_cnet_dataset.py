
import matplotlib.pyplot as plt
import numpy as np
from cnet_riff_dataset import CnetRiffDataset, preprocess_batch

# parameters

# names of files to segment
audio_files = ['pop.00000.wav', 'pop.00003.wav']
# folder where raw audio files are located in
audio_files_dir = "../pop-data/"
# where to output everything to
dataset_dir = "test_dataset_full_pipeline"

preprocess_batch(audio_files = audio_files,
                 audio_files_dir = audio_files_dir,
                 output_dir = dataset_dir,
                 fs=22050,
                 verbose=True,
                 save_wav=True)

# collect all training data into training object
train_dataset = CnetRiffDataset(dataset_dir)

# show sample contents if desired
print("Sample contents of dataset")
item = train_dataset[0]
plt.imshow((item['jpg'] + 1 )/ 2)
plt.title("Target spectrogram")
plt.figure()
plt.imshow(item['hint'])
plt.title("Source (canny edges)")
print("prompt:", item['txt'])