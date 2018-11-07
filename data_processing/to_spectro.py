from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import wave
import audioop
import os

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    #print("nperseg: " + str(nperseg))
    #print("noverlap: " + str(noverlap))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

#read in stereo file, write out mono file
def stereo_to_mono(infile, outfile):
    try:
        stereo = wave.open(infile, 'rb')
    except:
        stereo = wave.open(infile, 'r')
    mono = wave.open(outfile, 'wb')
    mono.setparams(stereo.getparams())
    mono.setnchannels(1)
    mono.writeframes(audioop.tomono(stereo.readframes(float('inf')), stereo.getsampwidth(), 1, 1))
    mono.close()

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
    #initialize variable to store samples (list we convert to npy array later)
    trainX = []
    #base for file names
    #base_num = 0
    #iterate over files
    m = 0 #batch size
    base_folder = "Dropbox/CS230/Mandarin_3secDataset_Stereo/"
    #for i in range(m):
    for stereo_file in os.listdir(base_folder):
        try:
            base_array = np.zeros((300,300))
            #ref_num = base_num + i
            stereo_file = base_folder + stereo_file
            print(stereo_file)
            mono_file = "Docs/Mandarin_mono/" + str(m) + ".wav"
            #convert to mono
            stereo_to_mono(stereo_file, mono_file)
            #read in audio file
            sample_rate, audio = wavfile.read(mono_file)
            #extract spectrogram
            _,_, spectrogram = log_specgram(audio, sample_rate)
            print(np.shape(spectrogram))
            base_array[0:299,0:81] = spectrogram
            trainX.append(base_array)
            m += 1
            print(m)
            #save backup copy
            if (batch_size % 10) == 0:
                trainX = np.array(trainX)
                np.save("Docs/audioNPY/trainX",trainX)
            print(m)
        except:
            print("error on: " + stereo_file)
            continue
    trainX = np.array(trainX)
    np.save("trainX",trainX)
    #to load: 
    #trainX = np.load("trainX.npy")
    #make a matching Y array
    #trainY = np.zeros((m,1))
    #trainY[:] = 0 #0 for english
    #print(np.shape(trainY))
    #np.save("trainY",trainY)






