import IPython.display as ipd
import librosa, librosa.display
import matplotlib.pyplot as plt
import pandas as pd
#from playsound import playsound

audio_sample = 'UrbanSound8K/audio/fold1/101415-3-0-2.wav'
plt.figure(figsize=(15,15))
data, sample_rate = librosa.load(audio_sample)
_ = librosa.display.waveplot(data, sr=sample_rate)
ipd.Audio(audio_sample)
#plt.show() #take out for jupyter notebook
#playsound(audio_sample) -> install with pip to hear on Ubuntu

data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
print(data.head())