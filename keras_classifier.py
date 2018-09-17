# Basic CNN (using Keras) following tutorial at
#     https://github.com/ajhalthor/audio-classifier-convNet
# Machine learning library: keras

import numpy as np # following all used by train_simple_keras
import keras 
from   keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from   keras.models import Sequential
from   keras.models import model_from_json
from   keras import backend as K 
import librosa
import librosa.display
import pandas as pd 
import random 
import warnings
warnings.filterwarnings('ignore')

import os
import sys 
sys.path.insert(0, '/home/dgabutler/Work/CMEEProject/Code')
import wavtools # contains custom functions e.g. denoising

from   pydub import AudioSegment # used by: search_file_
from   pydub.utils import make_chunks # used by: search_file_
from   shutil import copyfile # used by: search_file_
from   shutil import rmtree # used by: search_file_
import glob # used by: search_file_, search_folder_, hard_negative_miner
import subprocess # used by: search_file_
import time # used by: search_folder_
import pathlib # used by: search_file_
import csv # used by: search_file_
import matplotlib.pyplot as plt # used by train_simple_
from   sklearn.preprocessing import LabelEncoder # added for single output layer
from   sklearn.preprocessing import scale
from   sklearn.metrics import roc_curve # used in train_simple_
from   sklearn.metrics import auc

import functools    # these three used in as_keras_metric
from   keras import backend as K
import tensorflow as tf

def hard_negative_miner(wav_folder, threshold_confidence, model):

    # list all file names in folder
    wavs = glob.glob(wav_folder+'*.WAV')
    wavs = [os.path.splitext(x)[0] for x in wavs]
    wavs = [os.path.basename(x) for x in wavs]

    for wav in wavs:
        search_file_for_monkeys_ONE_NODE_OUTPUT(wav, threshold_confidence=threshold_confidence, wav_folder=wav_folder, sr=sr, model=model, hnm=True)

# ## NB.! time per file is approx. 1.553 seconds. 
# ## resulting in time per folder (~4000 files) of approx. 1.75 hours 


# ########################## REMOVE THIS SECTION WHEN I'VE STOPPED EXPERIMENTING WITH IT ####################################################
####### - added so functions and application of functions in same script, preventing having to import updated functions every time ##########

import os 
import sys
import csv 
import glob
import random
import pickle
import time

sys.path.insert(0, '/home/dgabutler/Work/CMEEProject/Code')
import wavtools   # contains custom functions e.g. denoising

praat_files = sorted(os.listdir('/home/dgabutler/Work/CMEEProject/Data/praat-files'))

# RUN 1 - no preprocessing
# compile dataset
# D_original = compile_dataset('without-preprocessing',sr=44100)

# RUN 2 - denoised only, no augmentations
# compile dataset
# D_denoised = compile_dataset('denoised',sr=44100)

# RUN 3 - denoised, Gaussian noise augmentation added
# compile dataset
# D_denoised_noise_aug = compile_dataset('denoised/noise-aug',sr=44100)

# RUN 4 - denoised, unbalanced classes (all known negatives)
# compile dataset
# D_denoised_noise_aug = compile_dataset('denoised/noise-aug',sr=44100)

# RUN 5 - denoised, random crop augmentation, unbalanced classes (all known negatives)
# compile dataset
# D_denoised_crop_aug_unbalanced = compile_dataset('denoised/crop-aug/unbalanced',sr=44100)

# training for a given run_type:
train_perc = 0.8
batch_size = 16
num_epochs = 10
sr = 44100
# dataset = D_original
name = 'D_original'
# train_simple_keras_ONE_NODE_OUTPUT(dataset,name,train_perc,num_epochs,batch_size,sr)
# train_simple_keras(dataset,name,train_perc,num_epochs,batch_size)

# BELOW WAS WORKFLOW BEFORE I CAME UP WITH COMPILE DATA FUNCTION,
# WHICH IS WHAT I USED ABOVE

# # # method 5: adding hard-negative mined training examples 

# # # # hard_negative_miner('/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/', 62, model=loaded_model)
# # # D_mined_aug_tb = D_aug_tb 
# # # wavtools.add_files_to_dataset(folder='mined-false-positives', dataset=D_mined_aug_tb, example_type=0)

# # # print("\nNumber of samples when hard negatives added: " + str(wavtools.num_examples(D_mined_aug_tb,0)) + \
# # # " negative, " + str(wavtools.num_examples(D_mined_aug_tb,1)) + " positive"))

# # # D_mined_aug_tb_denoised = wavtools.denoise_dataset(D_mined_aug_tb)

# # # method 6: adding selected obvious false positives as training examples

# # D_S_mined_aug_t_denoised = D_aug_t

# # wavtools.add_files_to_dataset(folder='selected-false-positives', dataset=D_S_mined_aug_t_denoised, example_type=0)

# # print("\nNumber of samples when select negatives added: " + str(wavtools.num_examples(D_S_mined_aug_t_denoised,0)) + \
# # " negative, " + str(wavtools.num_examples(D_S_mined_aug_t_denoised,1)) + " positive"))

# # # method 7: adding 'most wrong' false positives as training examples

# # # tried ~100 negatives from Catappa, ~100 positives that I had 
# # # DID NOT WORK. great results but background noise between positives and negatives was too different to generalise
# # # workflow was:
# # D_MW_mined = []
# # wavtools.add_files_to_dataset(folder='clipped-whinnies', dataset=D_MW_mined, example_type=1)
# # wavtools.add_files_to_dataset(folder='selected-false-positives/from-unclipped-whinnies', dataset=D_MW_mined, example_type=0)
# # wavtools.add_files_to_dataset(folder='selected-false-positives/catappa2-from-jenna', dataset=D_MW_mined, example_type=0)
# # D_MW_mined_denoised = wavtools.denoise_dataset(D_MW_mined)

# ### VIEWING SPECTROGRAMS 
# wav = '../Data/clipped-whinnies/5A3844FE_1.WAV'
# mag_spec = wavtools.load_mag_spec(wav, sr=41000, denoise=False, normalize=False)
# # different mel.spec generating methods
# mel_spec = D_denoised[24][0]
# mel_spec = wavtools.do_augmentation(mag_spec, sr=41000, noise=False, noise_samples=False, roll=False)

def view_mag_spec(mag_spec):
    librosa.display.specshow(librosa.power_to_db(mag_spec),x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Magnitude spectrogram')
    plt.show()

def view_mel_spec(mel_spec):
    librosa.display.specshow(librosa.power_to_db(mel_spec),y_axis='mel',x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()

def view_mag_and_mel(mag_spec, mel_spec):
    # mag
    plt.interactive(True)
    plt.subplot(1,2,1)
    librosa.display.specshow(librosa.power_to_db(mag_spec,ref=np.max),y_axis='linear',x_axis='time', sr=44100, )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Magnitude spectrogram')
    # mel
    plt.subplot(1,2,2)
    mel_spec = np.flip(mel_spec,axis=0)
    plt.imshow(mel_spec)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    wavname = os.path.basename(os.path.splitext(wav)[0])
    plt.savefig('../Results/viewing-input-spects/'+wavname+'.png')
    plt.gcf().clear()

# view_mag_and_mel(mag_spec, mel_spec)

# can I cut the input size...
# viewing all spects for the positives we have:
# folder = 'clipped-whinnies'
# data_folder_path = '../Data/'
# files = glob.glob(data_folder_path+folder+'/*.WAV')
# for wav in files:
#     mag_spec = wavtools.load_mag_spec(wav, sr=41000, denoise=False, normalize=False)
#     mel_spec = wavtools.do_augmentation(mag_spec, sr=41000, noise=False, noise_samples=False, roll=False)
#     view_mag_and_mel(mag_spec,mel_spec)

# AVERAGE CALL LENGTH OF ALL CALLS RECORDED

call_durations = []

for wav in praat_files:
    try:
	    start_times = wavtools.whinny_starttimes_from_praatfile('../Data/praat-files/'+wav)[1]
    except FileNotFoundError:
        print("Unable to process file:", wav)
        continue
    end_times = wavtools.whinny_endtimes_from_praatfile('../Data/praat-files/'+wav)[1]

    call_durations.extend([ends-starts for starts,ends in zip(start_times, end_times)])

avg_call_len = sum(call_durations) / float(len(call_durations))
longest_call = np.max(call_durations)
shortest_call = np.min(call_durations)

#################################################################
## MISC. CODE. NOT USED AT PRESENT

def custom_recall(y_true, y_pred, threshold):
    """Recall metric, with threshold option
    """
    y_true = np.ndarray.tolist(y_true)
    y_pred = np.ndarray.tolist(y_pred)
    # flatten list of lists to list
    y_pred = [item for sublist in y_pred for item in sublist]
    # round predictions to 0 or 1
    for idx, val in enumerate(y_pred):
        y_pred[idx] = 0 if val < threshold else 1 
    # true positives equals num. that sum to 2 ('1' and '1')
    summed = [x+y for x, y in zip(y_true, y_pred)]
    true_positives = sum(i == 2 for i in summed)
    # loop to find false positives
    possible_positives = sum(i == 1 for i in y_true)
    recall = true_positives/(true_positives+possible_positives)
    return recall 

def custom_precision(y_true, y_pred, threshold):
    """Precision metric, with threshold option
    """
    y_true = np.ndarray.tolist(y_true)
    y_pred = np.ndarray.tolist(y_pred)
    # flatten list of lists to list
    y_pred = [item for sublist in y_pred for item in sublist]
    # round predictions to 0 or 1
    for idx, val in enumerate(y_pred):
        y_pred[idx] = 0 if val < threshold else 1 
    # true positives equals num. that sum to 2 ('1' and '1')
    summed = [x+y for x, y in zip(y_true, y_pred)]
    true_positives = sum(i == 2 for i in summed)
    # loop to find false positives
    false_positives = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 0:
            false_positives += 1 
    precision = true_positives/(true_positives+false_positives)
    return precision 

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        Taken from https://stackoverflow.com/questions/42606207
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        Taken from: https://stackoverflow.com/questions/42606207
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

# following section taken from https://stackoverflow.com/questions/6392739
# in attempt to troubleshoot metric value problems (prev. too high)
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
@as_keras_metric
def auc_pr(y_true, y_pred, curve='PR'):
    return tf.metrics.auc(y_true, y_pred, curve=curve)


#################################################################
# MORE COMPLEX CONVNET WITH SPECIFIC IMPORT STATEMENTS
# from https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%202b%20-%20Train%20and%20Predict%20on%20UrbanSound%20dataset.ipynb

## NB! CURRENTLY TAKES MORE TIME AND IS LESS ACCURATE
## --- was abandoned early on so code may need edits to run 

# # change the seed before anything else
# import numpy as np
# np.random.seed(1)
# import tensorflow as tf
# tf.set_random_seed(1)

# import os
# import time

# import keras
# keras.backend.clear_session()

# import matplotlib.pyplot as plt
# import sklearn

# from keras.models import Sequential
# from keras.layers import Activation
# from keras.layers import Convolution2D, MaxPooling2D, Dropout
# from keras.layers.pooling import GlobalAveragePooling2D
# from keras.optimizers import Adamax
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.regularizers import l2
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# from keras.layers.normalization import BatchNormalization

# def train_complex_keras(dataset, train_perc):
#     """
#     Trains and more complex keras model. 

#     Taken from https://github.com/Azure/DataScienceVM/blob/master/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%202b%20-%20Train%20and%20Predict%20on%20UrbanSound%20dataset.ipynb

#     """
#     try:
#         random.shuffle(dataset)
#     except NameError:
#         print('non-existent dataset name provided. check dataset exists and retry')
#         return 

    # # use provided training percentage to give num. training samples
    # n_train_samples = int(round(len(dataset)*train_perc))
    # train = dataset[:n_train_samples]
    # # tests on remaining % of total
    # test = dataset[n_train_samples:]    

    # X_train, y_train = zip(*train)
    # X_test, y_test = zip(*test)

    # # reshape for CNN input
    # X_train = np.array([x.reshape( (128, 279, 1) ) for x in X_train])
    # X_test = np.array([x.reshape( (128, 279, 1) ) for x in X_test])

    # # one-hot encoding for classes
    # y_train = np.array(keras.utils.to_categorical(y_train, 2))
    # y_test = np.array(keras.utils.to_categorical(y_test, 2))

    # model = Sequential()
    # input_shape=(128, 279, 1)

    # # section 1

    # model.add(Convolution2D(filters=32, kernel_size=5,
    #                         strides=2,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal",
    #                         input_shape=input_shape))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=32, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.3))

    # # section 2    
    # model.add(Convolution2D(filters=64, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=64, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # # section 3
    # model.add(Convolution2D(filters=128, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=128, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=128, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(Convolution2D(filters=128, kernel_size=3,
    #                         strides=1,
    #                         padding="same",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    # # section 4
    # model.add(Convolution2D(filters=512, kernel_size=3,
    #                         strides=1,
    #                         padding="valid",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    # model.add(Convolution2D(filters=512, kernel_size=1,
    #                         strides=1,
    #                         padding="valid",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    # # section 5
    # model.add(Convolution2D(filters=2, kernel_size=1,
    #                         strides=1,
    #                         padding="valid",
    #                         kernel_regularizer=l2(0.0001),
    #                         kernel_initializer="normal"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(GlobalAveragePooling2D())

    # model.add(Activation('softmax'))

    # model.compile(
    #     optimizer="Adam",
    #     loss="binary_crossentropy",
    #     metrics=['accuracy'])

    # model.fit(
    #     x=X_train, 
    #     y=y_train,
    #     epochs=12,
    #     batch_size=128,
    #     validation_data= (X_test, y_test))

    # score = model.evaluate(
    #     x=X_test,
    #     y=y_test)

    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # serialise model to JSON
    # dataset_name = name
    # save_path = '/home/dgabutler/Work/CMEEProject/Models/'+dataset_name+'/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # model_json = model.to_json()
    # with open(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.json', 'w') as json_file:
    #     json_file.write(model_json)
    # # serialise weights to HDF5
    # model.save_weights(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.h5')
    # print('\nsaved model '+dataset_name+'/'+'e'+str(num_epochs)+'_b'+str(batch_size)+' to disk')
