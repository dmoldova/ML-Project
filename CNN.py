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
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from datetime import datetime

#Copy important code from Manipulation.py
csv_data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
audio_data = []
def extract_feat(name):#found files in csv that don't exist in subfolders; added try... except
	try:
		audio, sample = librosa.load(name, res_type='kaiser_fast') 
		mfccs = librosa.feature.mfcc(y = audio, sr = sample, n_mfcc = 40)
		pad = 174 - mfccs.shape[1]
		mfccs_scaled = np.pad(mfccs, pad_width((0,0), (0, pad_width)), mode = 'constant')
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

print("Split the data successfully")

# row = 40
# colmn = 174
# channel = 1

x_train = x_train.reshape(x_train.shape[0], 40, 174, 1)
x_test = x_test.reshape(x_test.shape[0], 40, 174, 1)

num_labels = final_categorical.shape[1]
filter_size = 2

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2, input_shape = (41, 174, 1), activation='relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = 2, activation='relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = 2, activation='relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = 2, activation='relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'], optimizer = 'adam')
model.summary()

score = model.evaluate(x_test, y_test, verbose = 1)
acc = 100 * score[1]

print("Pre-training accuracy: %.2f%%" % acc)