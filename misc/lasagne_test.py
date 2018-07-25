""" Practice in neural nets, theano, and lasagne """

__author__ = "Duncan Butler"
__date__ = "18.04.18"


## The following uses this website as tutorial:
# https://www.kdnuggets.com/2017/12/audio-classifier-deep-neural-networks.html
import librosa 
import glob
import numpy as np

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 128, frames = 128):
    window_size = 512*127
    log_specgrams = []
    labels = []
    ITJ = 0
    for l, sub_dir in enumerate(sub_dirs):
        PTJ = 1
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.logamplitude(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)
            print PTJ+ITJ*270,ITJ
            PTJ = PTJ+1
        ITJ = ITJ+1
    log_specgrams = np.array(log_specgrams)
    print log_specgrams.shape
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames)
    features = log_specgrams
    return np.array(features)



## THE FOLLOWING FROM THIS LINK, NOT CURRENTLY DOING IT

# import IPython.display as ipd
# ipd.Audio('../Sandbox/Urban_Sound_Data/train/Train/2022.wav')
# # Load data
# data, sampling_rate = librosa.load('../Sandbox/Urban_Sound_Data/train/Train/2022.wav')

