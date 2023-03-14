import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


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