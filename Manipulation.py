#Processing the data

import librosa, os, librosa.display
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.io import wavfile as wav
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

#What kind of data are we working with?
csv_data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
#print(data.classification.value_counts()) #show total number of audio samples in each class

audio_data = []

#convert all audio samples to 16-bit for wavfile if it's not 16- or 32-bit
# for root, dirs, files in os.walk("UrbanSound8K/audio/", topdown = True):
# 	for name in files:
# 		try: 
# 			filename = os.path.join(root, name)
# 			#print(filename)
# 			librosa_audio, librosa_sample_rate = librosa.load(filename)
# 			scipy_sample_rate, scipy_audio = wav.read(filename)
# 			#plt.figure(figsize = (15, 5))
# 			#plt.plot(scipy_audio)
# 			#print(scipy_sample_rate)
# 			#print(librosa_sample_rate)
# 		except ValueError as e:
# 			data, samplerate = sf.read(filename)
# 			sf.write(filename, data, samplerate, subtype='PCM_16')

def extract_feat(name):#found files in csv that don't exist in subfolders; added try... except
	try:
		audio, sample = librosa.load(name) 
		mfccs = librosa.feature.mfcc(y = audio, sr = sample, n_mfcc = 40)
		mfccs_scaled = np.mean(mfccs.T, axis = 0)
	except Exception:
		return None

	return mfccs_scaled

#Extract datatypes and features for CSV file
for ind, row in csv_data.iterrows(): 
	filename = os.path.join("UrbanSound8K/audio", "fold"+str(row["fold"]) + "/", str(row["slice_file_name"]))
	class_feat = row["classification"]
	data = extract_feat(filename)
	audio_data.append([data, class_feat])

features_dataframe = pd.DataFrame(audio_data, columns = ['features', 'classification'])
print("Parsed ", len(features_dataframe), " files")

final_feat_array = np.array(features_dataframe.features.tolist())
final_class_array = np.array(features_dataframe.classification.tolist())

final_categorical = to_categorical(LabelEncoder().fit_transform(final_class_array))

x_train, x_test, y_train, y_test = train_test_split(final_feat_array, final_categorical, test_size = 0.2, random_state = 42)

