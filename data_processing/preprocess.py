from scipy.io.wavfile import read
import numpy as np
import os

# A set of mu law encode/decode functions implemented
# in numpy
def mu_law_encode(audio):
	mu = 255
	audio = audio.astype('float32')
	range_signal = 33000.0
	#range_signal = max(audio) - min(audio)
	audio = audio / (range_signal * 1.0)
	magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
	audio = np.sign(audio) * magnitude
	audio = (audio + 1) / 2 * mu + 0.5
	quantized_signal = audio.astype(np.int32)
	return quantized_signal

def mu_law_decode(signal):
    # Calculate inverse mu-law companding and dequantization
    mu = 255
    y = signal.astype(np.float32)
    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu)**abs(y) - 1.0)
    return x

if __name__ == '__main__':
	data = []
	base_folder = "15K_German/"
	m = 0 #sample size
	max_signal = 0
	min_signal = 0
	for wav_file in os.listdir(base_folder):
		try:
			infile = read(base_folder + wav_file)
			sample_rate, sample_data = infile
			sample_length = len(sample_data)
			#we might want to try applying a conversion (like companding) to the data next
			if sample_length == 48000:
				data.append(list(sample_data))
				print(m)
				m += 1
			else:
				print("error on sample_length: " + str(sample_length))
			if m % 12000 == 0:
				print("testing")
				np_data = np.asarray(data) #will retain dtype from earlier
				#np_data.astype('float16')
				np.save("german_"+str(m)+".npy", np_data)
				break
		except:
			continue


	#create y dset
	#y = np.zeros((m,1))
	#y[0:a] = 0 #for each class
	#y.astype(int)
	#np.save("name",y)
