import torch
import numpy as np
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