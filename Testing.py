import librosa, os, librosa.display
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.io import wavfile as wav
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import to_categorical, np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from datetime import datetime

#Copy important code from Manipulation.py
csv_data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
audio_data = []
def extract_feat(name):#found files in csv that don't exist in subfolders; added try... except
	try:
		audio, sample = librosa.load(name) 
		mfccs = librosa.feature.mfcc(y = audio, sr = sample, n_mfcc = 40)
		mfccs_scaled = np.mean(mfccs.T, axis = 0)
	except Exception:
		return None

	return mfccs_scaled

for ind, row in csv_data.iterrows(): 
	filename = os.path.join("UrbanSound8K/audio", "fold"+str(row["fold"]) + "/", str(row["slice_file_name"]))
	class_feat = row["classification"]
	data = extract_feat(filename)
	audio_data.append([data, class_feat])

features_dataframe = pd.DataFrame(audio_data, columns = ['features', 'classification'])
print("Parsed ", len(features_dataframe), " files")

final_feat_array = np.array(features_dataframe.features.tolist())
final_class_array = np.array(features_dataframe.classification.tolist())
le = LabelEncoder()
final_categorical = to_categorical(le.fit_transform(final_class_array))

x_train, x_test, y_train, y_test = train_test_split(final_feat_array, final_categorical, test_size = 0.2, random_state = 42)

number_of_labels = final_categorical.shape[1]
seq_model = Sequential()

seq_model.add(Dense(256, input_shape=(40,)))
seq_model.add(Activation('relu'))
seq_model.add(Dropout(0.5))
seq_model.add(Dense(256))
seq_model.add(Activation('relu'))
seq_model.add(Dropout(0.5))
seq_model.add(Dense(number_of_labels))
seq_model.add(Activation('softmax'))

seq_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# seq_model.summary()

# score = seq_model.evaluate(x_test, y_test)
# accuracy = 100 * score[1]
# print("Pre-training accuracy: ", round(accuracy, 2), "%")

start = datetime.now()
seq_model.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_test, y_test), verbose = 1)
duration = datetime.now() - start
print("Training complete in ", duration, 2)

score = seq_model.evaluate(x_train, y_train, verbose = 0)
print("Training Accuracy: ", score[1])

score = seq_model.evaluate(x_test, y_test, verbose = 0)
print("Testing accuracy: ", score[1])