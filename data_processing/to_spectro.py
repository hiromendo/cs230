from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=5, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

#Saves a spectogram as an image
def save_spectro_as_img(spectrogram, file_name): #takes a spectrogram array
    #set up plt
    fig = plt.figure(frameon=False)
    #suppress axis
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #plot spectrogram
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    #save as image file
    fig.savefig(file_name + "spectro.png")


if __name__ == '__main__':
    #initialize variables
    trainX = []
    #put spectrogram into array
    for file in file_names:
        #read in audio file
        sample_rate, audio = wavfile.read(file + '.wav')
        #extract spectrogram
        _,_, spectrogram = log_specgram(audio, sample_rate)
        print(np.shape(spectrogram))
        #add assert size line when we know size here
        spectro_flat = spectrogram.reshape([])
        trainX.append(spectrogram)
    np.save("trainX",trainX)





