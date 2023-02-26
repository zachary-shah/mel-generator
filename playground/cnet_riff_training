
import numpy as np
import librosa
import os
import sys

sys.path.append('../ControlNet/')

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from cnet_riff_dataset import CnetRiffDataset
from ControlNet import tool_add_control


# TODO: add control to riffusion and svae to resume_path

# Configs
resume_path = './models/control_sd15_ini.ckpt'

batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# load dataset. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# load in dataset
dataset_dir = "test_dataset"
dataset = CnetRiffDataset(dataset_dir)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

# Train!
trainer.fit(model, dataloader)