from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#read in audio file
file_name = "goo"
sample_rate, audio = wavfile.read(file_name + '.wav')

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

#set up plt
fig = plt.figure(frameon=False)
#suppress axis
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
#extract spectrogram
_,_, spectrogram = log_specgram(audio, sample_rate)
#plot spectrogram
plt.imshow(spectrogram.T, aspect='auto', origin='lower')
#save as image file
fig.savefig(file_name + "spectro2.png")
