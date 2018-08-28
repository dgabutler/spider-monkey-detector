from   __future__ import print_function
import glob 
import librosa
import random 
import numpy as np
import keras 
from   keras.layers import Activation, Dense, Dropout, Conv2D, \
                           Flatten, MaxPooling2D, BatchNormalization
from   keras.models import Sequential

from   hyperopt import Trials, STATUS_OK, tpe
from   hyperas import optim
from   hyperas.distributions import choice, uniform
from   sklearn.preprocessing import LabelEncoder
from   keras import backend as K

import essential

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

    run_type = 'standardised'
    sr = 48000
    train_perc = 0.9

    if sr == 48000:
        time_dimension = 282
    if sr == 44100:
        time_dimension = 259

    x_train, y_train, x_test, y_test = essential.compile_dataset(run_type, sr)

    # reshape for CNN input
    x_train = np.array([x.reshape((128, time_dimension, 1)) for x in x_train])
    x_test = np.array([x.reshape((128, time_dimension, 1)) for x in x_test])

    # encoded 
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder.fit(y_test)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

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
    model.add(Activation({{choice(['relu', 'elu'])}}))
    
    if ({{choice(['batchnorm', 'without'])}}) == 'batchnorm': 
        model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation({{choice(['relu', 'elu'])}})) # relu

    if ({{choice(['batchnorm', 'without'])}}) == 'batchnorm': 
        model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation({{choice(['relu', 'elu'])}})) # relu

    if ({{choice(['batchnorm', 'without'])}}) == 'batchnorm': 
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout({{uniform(0, 1)}})) # 0.6152916582980337

    model.add(Dense({{choice([64, 128])}})) # 64
    model.add(Activation({{choice(['relu', 'elu'])}})) # relu
    model.add(Dropout({{uniform(0, 1)}})) # 0.23855852860918042

    if ({{choice(['batchnorm', 'without'])}}) == 'batchnorm': 
        model.add(BatchNormalization())

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}}) # 10**-2
    sgd = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}}) # 10**-1
    
    choiceval = {{choice(['adam', 'sgd'])}} # adam
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
        epochs=40, # 50
        batch_size={{choice([8, 16, 32])}}, # 32
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
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=30,
                                        trials=Trials())
    x_train, y_train, x_test, y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)