from   __future__ import print_function
import glob 
import librosa
import random 
import numpy as np
import keras 
from   keras.layers import Activation, Dense, Dropout, Conv2D, \
                           Flatten, MaxPooling2D, LeakyReLU
from   keras.models import Sequential

from   hyperopt import Trials, STATUS_OK, tpe
from   hyperas import optim
from   hyperas.distributions import choice, uniform


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    TRAIN_PERC = 0.8 
    FOLDERS = ['clipped-whinnies', 'clipped-negatives']

    data_folder_path = '/home/dgabutler/CMEECourseWork/Project/Data/'

    dataset = []
    i = 0   # counter distinguishes between folder 1 label and folder 2

    for folder in FOLDERS:
        files = glob.glob(data_folder_path+folder+'/*.WAV')
        for wav in files:
            if i == 0:
                label = 1
            else:
                label = 0
            y, sr = librosa.load(wav, sr=None, duration=3.00)
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
            if ps.shape != (128, 282): continue
            dataset.append((ps, label))
        i+=1    # increases for second folder, changing label 
                # from positive ('1') to negative ('0')
    
    random.shuffle(dataset)
    # use provided training percentage to give num. training samples
    n_train_samples = int(round(len(dataset)*TRAIN_PERC))
    train = dataset[:n_train_samples]
    # tests on remaining % of total
    test = dataset[n_train_samples:]    

    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)

    # reshape for CNN input
    x_train = np.array([x.reshape( (128, 282, 1) ) for x in x_train])
    x_test = np.array([x.reshape( (128, 282, 1) ) for x in x_test])

    # one-hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, 2))
    y_test = np.array(keras.utils.to_categorical(y_test, 2))

    return x_train, y_train, x_test, y_test

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
    model = Sequential()
    input_shape=(128, 282, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}})) # relu

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}})) # relu

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation({{choice(['relu', 'sigmoid'])}})) # relu

    model.add(Flatten())
    model.add(Dropout({{uniform(0, 1)}})) # 0.6152916582980337

    model.add(Dense({{choice([64, 128])}})) # 64
    model.add(Activation({{choice(['relu', 'sigmoid'])}})) # relu
    model.add(Dropout({{uniform(0, 1)}})) # 0.23855852860918042

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four': # three
        model.add(Dense({{choice([32, 64])}}))
        model.add(Activation({{choice(['relu', 'sigmoid'])}})) # sig
        model.add(Dropout({{uniform(0, 1)}})) # 0.055135448545175963

    model.add(Dense(2))
    model.add(Activation('softmax'))

    adam = keras.optimizers.Adam(lr={{choice([10**-4, 10**-3, 10**-2, 10**-1])}}) # 10**-2
    sgd = keras.optimizers.SGD(lr={{choice([10**-4, 10**-3, 10**-2, 10**-1])}}) # 10**-1
    rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-4, 10**-3, 10**-2, 10**-1])}}) # 10**-2
    
    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}} # adam
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd

    model.compile(
        optimizer=optim,
        loss="binary_crossentropy",
        metrics=['accuracy']) 

    model.fit(
        x=x_train, y=y_train,
        epochs={{choice([20, 30, 50])}}, # 50
        batch_size={{choice([8, 16, 32, 64])}}, # 32
        verbose=2, 
        validation_data= (x_test, y_test))

    score, acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=30,
                                        trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)