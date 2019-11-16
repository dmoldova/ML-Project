#Processing the data

import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile as wav

#What kind of data are we working with?
data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
print(data.classification.value_counts())