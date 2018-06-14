import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack
import numpy as np
from SamplePreProcessor import pre_process


def extract_features(sample_path):
    sample_rate, signal = wavfile.read(sample_path)
    signal = pre_process(sample_rate, signal)

    frame_size = 0.025  # 25 ms
    frame_stride = 0.01  # 10 ms

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]  # Some slicing I don't get

    frames *= np.hamming(frame_length)

    NFFT = 2048
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    num_ceps = 12
    cep_lifter = 22
    mfcc = fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    filter_banks_norm = filter_banks - (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc_norm = mfcc - (np.mean(mfcc, axis=0) + 1e-8)

    if __name__ == "__main__":
        axes[0].plot(signal)
        axes[1].imshow(np.flipud(filter_banks_norm.T), cmap='jet', aspect='auto',
                       extent=[0, signal_length / sample_rate, 0, sample_rate / 2000])
        # axes[1].imshow(np.flipud(mfcc_norm.T), cmap='jet', aspect='auto', extent=[0, num_ceps, 0, sample_rate / 2000])

    return filter_banks_norm.flatten(), mfcc_norm.flatten()


if __name__ == "__main__":
    for i in range(5):
        fig, axes = plt.subplots(1, 2)
        extract_features("Datasets//Lys/enreg-0{}.wav".format(i+1))
    plt.show()
