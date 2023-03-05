
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

from huggingface_hub import hf_hub_download

cntrl_riff_path = "pretrained_models/control_riffusion_ini.ckpt"

# get path of riffusion model
riffusion_path = hf_hub_download(repo_id="riffusion/riffusion-model-v1", filename="riffusion-model-v1.ckpt")

# add control to riffusion and save to resume_path
tool_add_control(riffusion_path, cntrl_riff_path)

# Configs
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True # MAY WANT TO TRY CHANGING THIS TO FALSE. then lower LR to 2e-6
only_mid_control = False

# load dataset. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(cntrl_riff_path, location='cpu'))
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