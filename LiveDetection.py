import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import time


p = pyaudio.PyAudio()
fs = 44100
recording_time = 0.1
samples_chunk = int(fs*recording_time)
chunks_displayed = int(1/recording_time)

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fs,
                input=True,
                frames_per_buffer=samples_chunk)

plt.ion()
fig, axes = plt.subplots(1, 1)
axes.set_ylim(-5000, 5000)
x = np.linspace(0, int(samples_chunk*chunks_displayed), int(samples_chunk*chunks_displayed))
line1, = axes.plot(x, np.zeros(int(samples_chunk*chunks_displayed)))

frames = np.zeros(int(samples_chunk*chunks_displayed), dtype=np.int16)
while 1:
    audio = np.fromstring(stream.read(samples_chunk), np.int16)
    frames = np.append(frames, audio)
    frames = frames[samples_chunk:]
    line1.set_ydata(frames)
    fig.canvas.draw()
    fig.canvas.flush_events()
