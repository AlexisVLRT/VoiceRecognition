import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from scipy import fftpack, interpolate
import numpy as np


def f_to_mel(f_domain_repr, f):
    mel_domain_repr = []
    for i in range(f_domain_repr):
        f = i*0
        mel_domain_repr.append(2595*np.log10(1+f/700))


def gen_filter(start, end, n):
    interp = interpolate.interp1d([0, 0.5, 1], [0, 1, 0])
    fraction = end-start
    new_scale = np.linspace(0, 1, n*fraction)
    return np.concatenate((np.zeros(int(start*n)), interp(new_scale), np.zeros(int((1-end)*n))))

audio = []
for dir in os.listdir("Datasets//Single//"):
    audio.append(wavfile.read("Datasets//Single//" + dir)[1])

fs = 44100
f = [np.linspace(0, fs/2, len(wave)//2) for wave in audio]
mel = [[2595*np.log10(1+f_sample/700) for f_sample in single_f] for single_f in f]
f_domain = [abs(fftpack.fft(wave))[:len(wave)//2] for wave in audio]
nfft = [len(spectrum) for spectrum in f_domain]
power_spectrum = [f_domain[i]**2/nfft[i] for i in range(len(nfft))]
filters = [[gen_filter(i/40, (i+1)/40, nfft_single) for i in range(40)] for nfft_single in nfft]

fig, axes = plt.subplots(5, 1)
[axes[0].plot(wave) for wave in audio]
[axes[1].plot(f[i][:5500], f_domain[i][:5500]) for i in range(len(f_domain))]
[axes[2].plot(mel[i][:5500], f_domain[i][:5500]) for i in range(len(f_domain))]
[axes[3].plot(mel[i][:5500], power_spectrum[i][:5500]) for i in range(len(f_domain))]
plt.show()