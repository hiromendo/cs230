from scipy.io.wavfile import read
import numpy as np
import os

data = []
base_folder = "Mandarin_Mono_3sec/"
m = 0 #sample size
for wav_file in os.listdir(base_folder):
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
		print(wav_file)
	if m % 500 == 0:
		np_data = np.asarray(data) #will retain dtype from earlier
		np.save("samples_"+str(m)+".npy", np_data)

#create y dset
#y = np.zeros((m,1))
#y[0:a] = 0 #for each class
#y.astype(int)
#np.save("name",y)







