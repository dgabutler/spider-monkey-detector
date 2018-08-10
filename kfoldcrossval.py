# script runs 10-fold cross validation on classifier

import glob 
import librosa
import random 
import numpy as np
import keras 
import operator 
from   keras.layers import Activation, Dense, Dropout, Conv2D, \
                           Flatten, MaxPooling2D
from   keras.models import Sequential
from   keras import backend as K
from   sklearn.preprocessing import LabelEncoder 
from   sklearn.model_selection import StratifiedKFold


import sys 
sys.path.insert(0, '/home/dgabutler/Work/CMEEProject/Code')
import wavtools # contains custom functions e.g. denoising

def data(run_type, train_perc, sr):
    """
    Data providing function.
    Run type argument specifies which set of data to load, 
    e.g. augmented, denoised.
    Current option is:
    - 'denoised/standardised'
    Also function has train percentage argument, to split
    the data into a given train/test proportion.
    """
    # increased sampling rate gives increased number of time-slices per clip
    # this affects CNN input size, and time_dimension used as proxy to ensure 
    # all clips tested are of the same length (if not, they are not added to test dataframe)
    if sr == 44100:
        time_dimension = 259
    elif sr == 48000:
        time_dimension = 282
    else:
        return("error: sampling rate must be 48000 or 44100") 
    # run-type optional processing methods
    if run_type == 'denoised/standardised':
        FOLDERS = ['clipped-whinnies', 'clipped-negatives']
        data_folder_path = '/home/dgabutler/Work/CMEEProject/Data/'
        dataset = []
        i = 0   # counter distinguishes between folder 1 label and folder 2
        for folder in FOLDERS:
            files = glob.glob(data_folder_path+folder+'/*.WAV')
            for wav in files:
                if i == 0:
                    label = 1
                else:
                    label = 0
                y, sr = librosa.load(wav, sr=sr, duration=3.00)
                ps = librosa.feature.melspectrogram(y=y, sr=sr)
                if ps.shape != (128, time_dimension): continue
                dataset.append((ps, label))
            i+=1    # increases for second folder, changing label 
                    # from positive ('1') to negative ('0')
        dataset = wavtools.denoise_dataset(dataset)
        dataset = wavtools.standardise_inputs(dataset)
    random.shuffle(dataset)
    # use provided training percentage to give num. training samples
    n_train_samples = int(round(len(dataset)*train_perc))
    train = dataset[:n_train_samples]
    # tests on remaining % of total
    test = dataset[n_train_samples:]    
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    # reshape for CNN input
    x_train = np.array([x.reshape((128, time_dimension, 1)) for x in x_train])
    x_test = np.array([x.reshape((128, time_dimension, 1)) for x in x_test])
    # labelling
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder.fit(y_test)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    return x_train, y_train, x_test, y_test

def compile_model():
    """
    Model providing function.
    """
    model = Sequential()
    input_shape=(128, 259, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.6152916582980337)) # from hyperas

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.23855852860918042)) # from hyperas

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=['accuracy', precision, recall]) 

    return model 

def kfoldcrossval(run_type, sr): 
    X_train, Y_train, X_test, Y_test = data(run_type, 0.8, sr)
    X = np.concatenate((X_train,X_test), axis=0)
    Y = np.concatenate((Y_train,Y_test), axis=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # initialise lists of metrics 
    acc_list, loss_list, precision_list, recall_list = ([] for i in range(4))
    for index, (train, test) in enumerate(kfold.split(X, Y)):
        print('Training on fold ' + str(index+1) + '/10...')
        # create and compile model
        model = compile_model()
        # fit model
        train_history = model.fit(X[train], Y[train], epochs=10, batch_size=32, validation_data=(X[test],Y[test]), verbose=0)
        # evaluate model
        def metric_max(metric):
            """returns best (highest/lowest) metric score from training run 
            plus index (i.e. epoch) of the maximum value
            """
            max_index, max_value = max(enumerate(train_history.history[metric]), key=operator.itemgetter(1))
            return [max_value, max_index] 
        # append maximum metric score and epoch at which it was recorded
        acc_list.append(metric_max('val_acc'))
        precision_list.append((metric_max('val_precision')))
        recall_list.append(metric_max('val_recall'))
        # for loss metric, append loss value at last training epoch
        loss_list.append(train_history.history['val_loss'][-1])
        # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    print("\naccuracy: %.2f%% (+/- %.2f%%)" % (np.mean([item[0] for item in acc_list])*100, np.std([item[0] for item in acc_list])*100))
    print("loss: %.2f (+/- %.2f)" % (np.mean(loss_list), np.std(loss_list)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean([item[0] for item in precision_list])*100, np.std([item[0] for item in precision_list])*100))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean([item[0] for item in recall_list])*100, np.std([item[0] for item in recall_list])*100))

kfoldcrossval('denoised/standardised', sr=44100)