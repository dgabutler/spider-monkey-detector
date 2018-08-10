import glob 
import librosa
import random 
import numpy as np
import keras 
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                           Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras import backend as K
from sklearn.preprocessing import LabelEncoder 
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import sys 
sys.path.insert(0, '/home/dgabutler/Work/CMEEProject/Code')
import wavtools 

def data():
    """
    Data providing function.
    Run type argument specifies which set of data to load, 
    e.g. augmented, denoised.
    Current option is:
    - 'denoised/standardised'
    Also function has train percentage argument, to split
    the data into a given train/test proportion.
    """
    RUN_TYPE = 'denoised/standardised'
    TRAIN_PERC = 0.8
    SR = 44100
    if SR == 44100:
        time_dimension = 259
    elif SR == 48000:
        time_dimension = 282
    if RUN_TYPE == 'denoised/standardised':
        FOLDERS = ['clipped-whinnies', 'clipped-negatives']
        data_folder_path = '/home/dgabutler/Work/CMEEProject/Data/'
        dataset = []
        i = 0   
        for folder in FOLDERS:
            files = glob.glob(data_folder_path+folder+'/*.WAV')
            for wav in files:
                if i == 0:
                    label = 1
                else:
                    label = 0
                y, sr = librosa.load(wav, sr=SR, duration=3.00)
                ps = librosa.feature.melspectrogram(y=y, sr=sr)
                if ps.shape != (128, time_dimension): continue
                dataset.append((ps, label))
            i+=1    
        dataset = wavtools.denoise_dataset(dataset)
        dataset = wavtools.standardise_inputs(dataset)
    random.shuffle(dataset)
    n_train_samples = int(round(len(dataset)*TRAIN_PERC))
    train = dataset[:n_train_samples]
    test = dataset[n_train_samples:]    
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    x_train = np.array([x.reshape((128, time_dimension, 1)) for x in x_train])
    x_test = np.array([x.reshape((128, time_dimension, 1)) for x in x_test])
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder.fit(y_test)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    return x_train, y_train, x_test, y_test

def recall(y_true, y_pred, threshold):
    """Recall metric, with threshold option
    """
    y_true = np.ndarray.tolist(y_true)
    y_pred = np.ndarray.tolist(y_pred)
    y_pred = [item for sublist in y_pred for item in sublist]
    for idx, val in enumerate(y_pred):
        y_pred[idx] = 0 if val < threshold else 1 
    summed = [x+y for x, y in zip(y_true, y_pred)]
    true_positives = sum(i == 2 for i in summed)
    possible_positives = sum(i == 1 for i in y_true)
    recall = true_positives/(true_positives+possible_positives)
    return recall 

def precision(y_true, y_pred, threshold):
    """Precision metric, with threshold option
    """
    y_true = np.ndarray.tolist(y_true)
    y_pred = np.ndarray.tolist(y_pred)
    y_pred = [item for sublist in y_pred for item in sublist]
    for idx, val in enumerate(y_pred):
        y_pred[idx] = 0 if val < threshold else 1 
    summed = [x+y for x, y in zip(y_true, y_pred)]
    true_positives = sum(i == 2 for i in summed)
    false_positives = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 0:
            false_positives += 1 
    precision = true_positives/(true_positives+false_positives)
    return precision 

def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    SR = 44100
    if SR == 44100:
        time_dimension = 259
    elif SR == 48000:
        time_dimension = 282

    model = Sequential()
    input_shape=(128, time_dimension, 1)

    model.add(Conv2D(24, {{choice([(3, 3),(5, 5),(7, 7)])}}, strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu')) 
    if conditional({{choice(['batchnorm', 'without'])}}) == 'batchnorm': 
        model.add(BatchNormalization)

    model.add(Conv2D(24, {{choice([(3, 3),(5, 5)])}}, padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu')) 
    if conditional({{choice(['batchnorm', 'without'])}}) == 'batchnorm': 
        model.add(BatchNormalization)

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu')) 
    if conditional({{choice(['batchnorm', 'without'])}}) == 'batchnorm': 
        model.add(BatchNormalization)

    model.add(Flatten())
    model.add(Dropout({{uniform(0, 1)}})) 

    model.add(Dense(64))
    model.add(Activation('relu')) 
    if conditional({{choice(['batchnorm', 'without'])}}) == 'batchnorm': 
        model.add(BatchNormalization)
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = keras.optimizers.Adam(lr={{choice([10**-4, 10**-3, 10**-2, 10**-1])}}) # 10**-2
    sgd = keras.optimizers.SGD(lr={{choice([10**-4, 10**-3, 10**-2, 10**-1])}}) # 10**-1
    
    choiceval = {{choice(['adam', 'sgd'])}} 
    if choiceval == 'adam':
        optim = adam
    else:
        optim = sgd

    model.compile(
        optimizer=optim,
        loss="binary_crossentropy",
        metrics=['accuracy']) 

    model.fit(
        x=x_train, y=y_train,
        epochs=1,
        batch_size={{choice([8, 16, 32, 64])}}, 
        verbose=2, 
        validation_data=(x_test, y_test))

    y_pred = model.predict(x_test)
    recall_score = recall(y_test, y_pred, 0.5) 
    print('Recall:', recall_score)
    return {'loss': -recall_score, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=2,
                                        trials=Trials())
    x_train, y_train, x_test, y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)