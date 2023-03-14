import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import hilbert

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as f


import numpy as np

import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

import soundfile
import ffmpeg

import noisereduce as nr
import os
import re
import soundfile as sf
import pyloudnorm as pyln
import cdpam

import sys


def plot_waveform(waveform, sample_rate, title="Waveform"):

    if waveform.type() == 'torch.DoubleTensor':
        waveform = waveform.numpy()

    waveform = waveform.reshape(1, -1)
    
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram"):

    if waveform.type() == 'torch.DoubleTensor':
        waveform = waveform.numpy()

    waveform = waveform.reshape(1, -1)

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show(block=False)


def envelope_distance(predicted_binaural, gt_binaural):
    #channel1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural))
    gt_env_channel1 = np.abs(hilbert(gt_binaural))
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))
    #sum the distance between two channels
    envelope_distance = channel1_distance
    return float(envelope_distance)

#Generates a list of spectrograms functions, given a list of window sizes
#hop_ratio - the ratio of the hop_length to the window size.
#if the hop_ratio is p in (0,1), then the proportion of overlap between windows is 1-p.
def gen_spectrogram_fcns(window_sizes, hop_ratio):
    
    spectrograms = []
    for window_size in window_sizes:
        spectrograms.append(
            T.Spectrogram(
                n_fft=window_size,
                win_length=None,
                hop_length=int(window_size*hop_ratio),
                center=True,
                pad_mode="reflect",
                power=2.0,
            )
        )
    return spectrograms

#mel spectrogram
#can experiment with other norms
def gen_mel_spectrogram_fcns(window_sizes, hop_ratio):
    log_spectrograms = []
    for window_size in window_sizes:
        log_spectrograms.append(
            T.MelSpectrogram(
                n_fft=window_size,
                win_length=None,
                hop_length=int(window_size*hop_ratio),
                center=True,
                pad_mode="reflect",
                power=2.0,
                n_mels = int(window_size/8)
            )
        )
    return log_spectrograms

#Weights -  how much each window size is weighted
#mel - scales frequency axis by mel scale (logarithmic)
#hop ratio = 1-(amount of overlap between window)
#loss_fcn = distance metric between spectrograms
#Can take in a stack of waveforms


def safe_log(x, eps=1e-7):
	"""Avoid taking the log of a non-positive number."""
	safe_x = torch.where(x <= eps, eps, x)
	return torch.log(safe_x)

    
def compute_spectral_loss(waveform1, waveform2, mel=False, log=False,
    window_sizes=[128, 256, 512, 1024, 2048], weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    hop_ratio=0.5, loss_fcn=nn.L1Loss()):
    
    assert len(window_sizes) == len(weights), "list of lambdas and window sizes must be the same length"

    #Generates a list of spectrogram functions
    if mel:
        spectrogram_fcns = gen_mel_spectrogram_fcns(window_sizes, hop_ratio)
    else:
        spectrogram_fcns = gen_spectrogram_fcns(window_sizes, hop_ratio)

    #Generates spectrograms for each window size and waveform
    spectrograms1 = [spec(waveform1) for spec in spectrogram_fcns]
    spectrograms2 = [spec(waveform2) for spec in spectrogram_fcns]

    if log==True:
        spectrograms1 = [safe_log(spec) for spec in spectrograms1]
        spectrograms2 = [safe_log(spec) for spec in spectrograms2]

        
    loss = 0
    for i in range(len(spectrograms1)):
        loss += weights[i]*loss_fcn(spectrograms1[i], spectrograms2[i])

    return loss

def compute_DiffImpact_loss(waveform1, waveform2, lambda1=0.5, lambda2=0.5):
    return (lambda1 * compute_spectral_loss(waveform1, waveform2) + lambda2*compute_spectral_loss(waveform1, waveform2, log=True)).item()



def compute_cdpam_loss(waveform1, waveform2,batch_size=4):
 
    waveform1_np = waveform1.numpy()
    waveform1_processed = np.round(waveform1_np.astype(float)*32768)
    waveform1_processed = np.float32(waveform1_processed)

    waveform2_np = waveform2.numpy()
    waveform2_processed = np.round(waveform2_np.astype(float)*32768)
    waveform2_processed = np.float32(waveform2_processed)

    print("Tensor Size:")
    print(waveform1_processed.shape)
    print(waveform2_processed.shape)

    length_1 = waveform1_processed.shape[1]
    length_2 = waveform2_processed.shape[1]
    num_recs = waveform2_processed.shape[0]

    sum = 0
    for i in range(0, num_recs, batch_size):
        tensor = loss_fn.forward(waveform1_processed[i:(i+batch_size)], waveform2_processed[i:(i+batch_size)])
        sum += torch.sum(tensor).item()
        del tensor
        torch.cuda.empty_cache()
        if i%100 == 0:
            print(i)


    average = sum/num_recs
    del waveform1_np
    del waveform2_np
    del waveform1_processed
    del waveform2_processed
    print(average)
    return average


