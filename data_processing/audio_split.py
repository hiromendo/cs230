from pydub import AudioSegment

def audio_split(audioinput, time):

	t1 = 0 * 1000 #Works in milliseconds
	t2 = time * 1000
	
	newAudio = AudioSegment.from_wav(audioinput)
	newAudio = newAudio[t1:t2]

	newAudio.export('sx82_short2.wav', format="wav")

audio = "sx82.wav"

audio_split(audio, 2)