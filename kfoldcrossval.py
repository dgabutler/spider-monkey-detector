# script runs 10-fold cross validation on classifier
# NB - CANNOT BE USED ON AUGMENTED DATA

# packages used
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
from   sklearn.utils import shuffle 
import os 
import pickle

# supresses 'exceding 10% memory allocation' warnings from tf 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# custom modules
import essential

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

    print "Dataset compiled"

    X = np.concatenate((x_train,x_test), axis=0)
    Y = np.concatenate((y_train,y_test), axis=0)

    return X, Y

# following metrics courtesy of Avcu, see https://github.com/keras-team/keras/issues/5400
def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred
def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def compile_model():
    """
    Model providing function.
    """
    model = Sequential()
    input_shape=(128, 282, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5)) # from hyperas

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5)) # from hyperas

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=['accuracy', precision, recall, f1]) 

    return model 

def kfoldcrossval(X, Y, sr, num_epochs):

    print "Beginning 10-fold cross-validation:"
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # initialise lists of metrics 
    acc_list, loss_list, precision_list, recall_list, f1_list = ([] for i in range(5))
    for index, (train, test) in enumerate(kfold.split(X, Y)):
        print('Training on fold ' + str(index+1) + '/10...')
        # create and compile model
        model = compile_model()       
        # fit model
        train_history = model.fit(X[train], Y[train], epochs=num_epochs, batch_size=16, validation_data=(X[test],Y[test]), verbose=2)
        # evaluate model
        def metric_max(metric):
            """returns best (highest/lowest) metric score from training run 
            plus index (i.e. epoch) of the maximum value
            """
            max_index, max_value = max(enumerate(train_history.history[metric]), key=operator.itemgetter(1))
            return [max_value, max_index] 
        
        # BELOW APPROACH CHANGED, max. should really be median
        # append maximum metric score and epoch at which it was recorded
        # acc_list.append(metric_max('val_acc'))
        # precision_list.append((metric_max('val_precision')))
        # recall_list.append(metric_max('val_recall'))
        # f1_list.append((metric_max('val_f1')))

        # MEDIAN - more representative
        acc_list.append(np.median(train_history.history['val_acc']))
        precision_list.append(np.median(train_history.history['val_precision']))
        precision_list.append(np.median(train_history.history['val_precision']))
        f1_list.append(np.median(train_history.history['val_f1']))
        # for loss metric, append loss value at last training epoch
        loss_list.append(train_history.history['val_loss'][-1])

    print("\nVal Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean([item[0] for item in acc_list])*100, np.std([item[0] for item in acc_list])*100))
    print("Val Loss: %.2f (+/- %.2f)" % (np.mean(loss_list), np.std(loss_list)))
    print("Val Precision: %.2f%% (+/- %.2f%%)" % (np.mean([item[0] for item in precision_list])*100, np.std([item[0] for item in precision_list])*100))
    print("Val Recall: %.2f%% (+/- %.2f%%)" % (np.mean([item[0] for item in recall_list])*100, np.std([item[0] for item in recall_list])*100))
    print("Val F1: %.2f%% (+/- %.2f%%)" % (np.mean([item[0] for item in f1_list])*100, np.std([item[0] for item in f1_list])*100))

    return acc_list, loss_list, precision_list, recall_list, f1_list

def sample_n_positives(n, X, Y):
    """
    Allows n random positives (with n random negatives) 
    to be drawn from dataset. Used to plot learning curve of network
    """
    neg_indices = np.ndarray.tolist(np.ndarray.flatten(np.flatnonzero(Y==0)))
    pos_indices = np.ndarray.tolist(np.ndarray.flatten(np.flatnonzero(Y==1)))

    sampled_neg_indices = random.sample(neg_indices, n)
    sampled_pos_indices = random.sample(pos_indices, n)

    sampled_neg_X = np.take(X, sampled_neg_indices, axis=0)
    sampled_pos_X = np.take(X, sampled_pos_indices, axis=0)

    sampled_neg_Y = np.take(Y, sampled_neg_indices)
    sampled_pos_Y = np.take(Y, sampled_pos_indices)

    X = np.concatenate((sampled_neg_X,sampled_pos_X), axis=0)
    Y = np.concatenate((sampled_neg_Y,sampled_pos_Y), axis=0)

    return X, Y 

#### OVERNIGHT RUN:
# (3) preprocessing:
print "kfold WITHOUT"
X, Y = data('without-preprocessing', 0.8, 48000)
without_acc_list, without_loss_list, without_precision_list, without_recall_list, without_f1_list = kfoldcrossval(X, Y, sr=48000, num_epochs=40)
print "kfold DENOISED"
X, Y = data('denoised', 0.8, 48000)
denoised_acc_list, denoised_loss_list, denoised_precision_list, denoised_recall_list, denoised_f1_list = kfoldcrossval(X, Y, sr=48000, num_epochs=40)
print "kfold STANDARDISED"
X, Y = data('standardised', 0.8, 48000)
stand_acc_list, stand_loss_list, stand_precision_list, stand_recall_list, stand_f1_list = kfoldcrossval(X, Y, sr=48000, num_epochs=40)
print "kfold DENOISED/STANDARDISED"
X, Y = data('denoised/standardised', 0.8, 48000)
denoised_stand_acc_list, denoised_stand_loss_list, denoised_stand_precision_list, denoised_stand_recall_list, denoised_stand_f1_list = kfoldcrossval(X, Y, sr=48000, num_epochs=40)

# (2) increasing sample size:
# lists to save results
# recall_means, recall_stds = ([] for i in range(2))
# precision_means, precision_stds = ([] for i in range(2))
# f1_means, f1_stds = ([] for i in range(2))

sample_sizes = [50, 75, 100, 124]

print "Compiling dataset:"
X, Y = data('standardised', 0.8, 48000)

# for size in sample_sizes:
#     testX, testY = sample_n_positives(size, X, Y)
#     print size
#     lc_acc_list, lc_loss_list, lc_precision_list, lc_recall_list, lc_f1_list = kfoldcrossval(X, Y, sr=48000, num_epochs=40)
    
#     recall_means.append(np.mean([item[0] for item in lc_recall_list])*100)
#     recall_stds.append(np.std([item[0] for item in lc_recall_list])*100)

#     f1_means.append(np.mean([item[0] for item in lc_f1_list])*100)
#     f1_stds.append(np.std([item[0] for item in lc_f1_list])*100)

#     precision_means.append(np.mean([item[0] for item in lc_precision_list])*100)
#     precision_stds.append(np.std([item[0] for item in lc_precision_list])*100)

# (1) AUGMENTATION vs. NO AUGMENTATION
# for no augmentation, see 'without' results above
print "kfold ALL AUG"
X, Y = data('all_aug', 0.8, 48000)
# all_aug_acc_list, all_aug_loss_list, all_aug_precision_list, all_aug_recall_list, all_aug_f1_list = kfoldcrossval(X, Y, sr=48000, num_epochs=40)

def summarise_WITHOUT_results():
    print "Val Recall: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in without_recall_list])*100, \
    np.std([item[0] for item in without_recall_list])*100)
    print "Val Precision: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in without_precision_list])*100, \
    np.std([item[0] for item in without_precision_list])*100)
    print "Val F1: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in without_f1_list])*100, \
    np.std([item[0] for item in without_f1_list])*100)
    print "Val Acc: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in without_acc_list])*100, \
    np.std([item[0] for item in without_acc_list])*100)
    print "Val Loss: %.2f (+/- %.2f)" % \
    (np.mean(without_loss_list), \
    np.std(without_loss_list))

def summarise_STAND_results():
    print "Val Recall: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in stand_recall_list])*100, \
    np.std([item[0] for item in stand_recall_list])*100)
    print "Val F1: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in stand_f1_list])*100, \
    np.std([item[0] for item in stand_f1_list])*100)
    print "Val Acc: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in stand_acc_list])*100, \
    np.std([item[0] for item in stand_acc_list])*100)
    print "Val Loss: %.2f (+/- %.2f)" % \
    (np.mean(stand_loss_list), \
    np.std(stand_loss_list))

def summarise_DENOISED_results():
    print "Val Recall: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in denoised_recall_list])*100, \
    np.std([item[0] for item in denoised_recall_list])*100)
    print "Val F1: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in denoised_f1_list])*100, \
    np.std([item[0] for item in denoised_f1_list])*100)
    print "Val Acc: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in denoised_acc_list])*100, \
    np.std([item[0] for item in denoised_acc_list])*100)
    print "Val Loss: %.2f (+/- %.2f)" % \
    (np.mean(denoised_loss_list), \
    np.std(denoised_loss_list))

def summarise_DENOISED_STAND_results():
    print "Val Recall: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in denoised_stand_recall_list])*100, \
    np.std([item[0] for item in denoised_stand_recall_list])*100)
    print "Val F1: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in denoised_stand_f1_list])*100, \
    np.std([item[0] for item in denoised_stand_f1_list])*100)
    print "Val Acc: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in denoised_stand_acc_list])*100, \
    np.std([item[0] for item in denoised_stand_acc_list])*100)
    print "Val Loss: %.2f (+/- %.2f)" % \
    (np.mean(denoised_stand_loss_list), \
    np.std(denoised_stand_loss_list))

def summarise_ALL_AUG_results():
    print "Val Recall: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in all_aug_recall_list])*100, \
    np.std([item[0] for item in all_aug_recall_list])*100)
    print "Val F1: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in all_aug_f1_list])*100, \
    np.std([item[0] for item in all_aug_f1_list])*100)
    print "Val Acc: %.2f%% (+/- %.2f%%)" % \
    (np.mean([item[0] for item in all_aug_acc_list])*100, \
    np.std([item[0] for item in all_aug_acc_list])*100)
    print "Val Loss: %.2f (+/- %.2f)" % \
    (np.mean(all_aug_loss_list), \
    np.std(all_aug_loss_list))

### SAVING RESULTS # 
a = all_aug_acc_list
a = [item[0] for item in a]
b = all_aug_loss_list
c = all_aug_precision_list
c = [item[0] for item in c]
d = all_aug_recall_list
d = [item[0] for item in d]
e = all_aug_f1_list
e = [item[0] for item in e]

WITHOUT = np.column_stack((a,b,c,d,e))
STAND = np.column_stack((a,b,c,d,e))
DENOISED = np.column_stack((a,b,c,d,e))
DENOISED_STAND = np.column_stack((a,b,c,d,e))
ALL_AUG = np.column_stack((a,b,c,d,e))

sample_sizes_RECALL = np.column_stack((recall_means,recall_stds))
sample_sizes_F1 = np.column_stack((f1_means,f1_stds))

np.save('../Results/WITHOUT.npy', WITHOUT)
np.save('../Results/STAND.npy', STAND)
np.save('../Results/DENOISED.npy', DENOISED)
np.save('../Results/DENOISED_STAND.npy', DENOISED_STAND)
np.save('../Results/ALL_AUG.npy', ALL_AUG)

np.save('../Results/sample_sizes_RECALL.npy', sample_sizes_RECALL)
np.save('../Results/sample_sizes_F1.npy', sample_sizes_F1)