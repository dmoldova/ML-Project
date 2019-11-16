#Processing the data

import librosa, os, librosa.display
import numpy as np
import pandas as pd
from scipy.io import wavfile as wav

#What kind of data are we working with?
data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
#print(data.classification.value_counts()) #show total number of audio samples in each class

#make sound samples mono
audio_data = []

for root, dirs, files in os.walk("UrbanSound8K/audio/", topdown = True):
	for name in files:
		filename = os.path.join(root, name)
		librosa_audio, librosa_sample_rate = librosa.load(filename)
		scipy_sample_rate, scipy_audio = wav.read(filename)
		print(scipy_sample_rate)
		print(librosa_sample_rate)


# for x in range(1,11):
# 	for filename in ('UrbanSound8K/audio/fold') + str(x) + '/*.wav':
# 		librosa_audio, librosa_sample_rate = librosa.load(filename)
# 		scipy_sample_rate, scipy_audio = wav.read(filename)
# 		print(scipy_sample_rate)
# 		print(librosa_sample_rate)