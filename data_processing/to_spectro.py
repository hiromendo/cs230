from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import wave
import audioop
import os
#from pydub import AudioSegment

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
    m = 0 #batch size
    errors = 0
    #base for file names
    base_folder = "Mandarin_Mono_3sec/"
    #iterate over files
    for stereo_file in os.listdir(base_folder):
        try:
            #set up padding array
            base_array = np.zeros((300,300))
            #set up file names
            stereo_file = base_folder + stereo_file
            mono_file = "mandarin_mono/" + str(m) + ".wav"
            #convert to mono
            stereo_to_mono(stereo_file, mono_file)
            #read in audio file
            sample_rate, audio = wavfile.read(mono_file)
            #extract spectrogram
            _,_, spectrogram = log_specgram(audio, sample_rate)
            os.remove(mono_file) #save memory by removing the mono file
            #print(np.shape(spectrogram))
            #pad spectrogram
            h,w = np.shape(spectrogram)
            base_array[0:h,0:w] = spectrogram
            #add to storage
            trainX.append(base_array)
            m += 1
            #save backup copy
            if (m % 10000) == 0:
                trainX_array = np.array(trainX)
                np.save("trainX_" + str(m),trainX_array)
                print(m)
        except:
            errors += 1
            print("error on: " + stereo_file + " total errors: " + str(errors))
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






