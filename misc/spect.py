import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('/home/dgabutler/Work/CMEEProject/Data/clipped-whinnies/5A3AD5B6_1.WAV', duration=3.00, sr=None)
D = librosa.stft(y)

ps = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.show()

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='linear', x_axis='time')
plt.title('Power spectrogram')
plt.show()

import glob 
import os
import numpy as np

#Iterates over folder of .WAV files, creating spectrograms of each
def specs_from_folder(folder_name):
    """
    Iterates over folder of .WAV files, creating spectrograms of each
    """
    folder_path = '/home/dgabutler/Work/CMEEProject/Data/'+folder_name
    wavs = glob.glob(folder_path+'/*.WAV')
    for wav in wavs:
        y, sr = librosa.load(wav, duration=3.00, sr=None)
        D = librosa.stft(y)
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='linear', x_axis='time')
        file_name = os.path.basename(os.path.splitext(wav)[0])
        plt.savefig('/home/dgabutler/Work/CMEEProject/Data/clipped-whinny-spectrograms/'+file_name+'.png')