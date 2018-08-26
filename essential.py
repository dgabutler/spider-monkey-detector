# only essential scripts to get version working

import numpy as np # following all used by train_simple_keras
import keras 
from   keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, BatchNormalization
from   keras.models import Sequential
from   keras.models import model_from_json
from   keras import backend as K 
import librosa
import librosa.display
import pandas as pd 
import random 
import warnings
warnings.filterwarnings('ignore')

import os, os.path
import sys

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

# added during experimentation
import numpy as np 
from   sklearn.utils import shuffle # used by do_augmentation
from   sklearn.metrics import precision_recall_curve

# custom modules
import wavtools # contains custom functions e.g. denoising

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def add_n_files_to_dataset(n, folder, dataset, example_type, sr, noise=False, noise_samples=False):
    """
    Takes all wav files from given folder name (minus slashes on 
    either side) and adds to the dataset name provided.
    Example type = 0 if negative, 1 if positive.
    sr (sampling rate) must be 44100 or 48000. 

    Also performs augmentations on mel spectrograms if conditionally 
    requested.

    n = -1 adds ALL files from folder to dataset,
    else only n files are randomly selected and added
    """
    data_folder_path = '/home/dgabutler/Work/CMEEProject/Data/'
    
    if n == -1:
        files = glob.glob(data_folder_path+folder+'/*.WAV')
    else: 
        all_files = glob.glob(data_folder_path+folder+'/*.WAV')
        random.shuffle(all_files)
        files = all_files[0:n]

    AUG = {#'type':[probability, value]
    'noise':[1, 0.01], # prob. was 0.1, now changed to 1
    'noise_samples':[1, 1.0] # prob. was 0.1, now changed to 1
    }

    RANDOM = np.random.RandomState(100)

    for wav in files:
        # print wav
        y, sr = librosa.load(wav, sr=sr, duration=3.00)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # augmentations:       
        # (1) gaussian noise
        if 'noise' and RANDOM.choice([True, False], p=[AUG['noise'][0], 1 - AUG['noise'][0]]):
            mel_spec = mel_spec + RANDOM.normal(0.0, RANDOM.uniform(0, AUG['noise'][1]**0.5), mel_spec.shape)
            mel_spec = np.clip(mel_spec, 0.0, 1.0)

        # (2) mixed with another noise e.g. howler monkey, chirping bird
        # load noise samples
        NOISE = shuffle(glob.glob('../Data/non-monkey-noises/*.WAV'))
        # add noise samples
        if 'noise_samples' and RANDOM.choice([True, False], p=[AUG['noise_samples'][0], 1 - AUG['noise_samples'][0]]):
            noise_samples, sr = librosa.load(NOISE[RANDOM.choice(range(0, len(NOISE)))], sr=sr, duration=3.00)
            noise_spec = librosa.feature.melspectrogram(y=noise_samples, sr=sr)
            mel_spec = mel_spec + (noise_spec * AUG['noise_samples'][1])
            # rescale combined image (preventing background noise levels from being doubled)
            mel_spec -= mel_spec.min(axis=None)
            mel_spec /= mel_spec.max(axis=None)

        # checking input size:    
        if sr == 48000:
            if mel_spec.shape != (128, 282): continue
        elif sr == 44100:
            if mel_spec.shape != (128, 259): continue
        else:
            print("error: sampling rate must be 48000 or 44100 - file", wav, "has sampling rate", sr) 
            continue 
        dataset.append((mel_spec, example_type))

    return dataset 

def denoise(spec_noisy):
    """
    Subtract mean from each frequency band
    Modified from Mac Aodha et al. 2017
    NB! load_mag_spec also has denoising element, 
    from Stefan Kahl on answer to stack overflow Q
    """
    me = np.mean(spec_noisy, 1)
    spec_denoise = spec_noisy - me[:, np.newaxis]

    # remove anything below 0
    spec_denoise.clip(min=0, out=spec_denoise)

    return spec_denoise

def denoise_dataset(dataset):
    """
    Applies denoise function over all spectrograms in dataset.
    Different functionality dependent on labelled/unlabelled data
    """
    if type(dataset[0]) == tuple:
        for spectrogram in dataset:
            spect_tuple_as_list = list(spectrogram)
            spect_tuple_as_list[0] = denoise(spect_tuple_as_list[0])
            spectrogram = tuple(spect_tuple_as_list)
    else: 
        for spectrogram in dataset:
            spectrogram = denoise(spectrogram)
    return dataset 

def standardise_dataset(dataset):
    """uses standardisation method from 
    https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    , applies to all spects in dataset
    """
    for row in dataset:
        row = list(row)
        row[0] /= np.max(np.abs(row[0]),axis=None)

    return dataset

def compile_dataset(run_type, sr):
    """
    Data providing function.
    Run type argument specifies which set of data to load, 
    e.g. augmented, denoised.
    Current options are:
    - 'without-preprocessing'
    - 'standardised'
    - 'denoised'
    - 'denoised/standardised'
    - 'crop_augmented'
    - 'all-aug'
    Returns dataset for input into CNN training function
    """
    # build standard dataset:
    def add_basic_dataset(dataset, noise, noise_samples):

        # positives: four locations (osa, catappa, shady lane, corcovado)
        print "...basic data: positives"
        dataset = add_n_files_to_dataset(n=-1, folder='clipped-whinnies-osa', dataset=dataset, example_type=1, sr=sr, noise=noise, noise_samples=noise_samples)
        dataset = add_n_files_to_dataset(n=-1, folder='clipped-whinnies-catappa', dataset=dataset, example_type=1, sr=sr, noise=noise, noise_samples=noise_samples)
        dataset = add_n_files_to_dataset(n=-1, folder='clipped-whinnies-shady', dataset=dataset, example_type=1, sr=sr, noise=noise, noise_samples=noise_samples)
        dataset = add_n_files_to_dataset(n=-1, folder='clipped-whinnies-corcovado', dataset=dataset, example_type=1, sr=sr, noise=noise, noise_samples=noise_samples)
        # negatives: three locations   
        # Osa:
        # - same num. osa negatives as osa positives
        print "...basic data: negatives"
        osa_n_pos = len(glob.glob('../Data/clipped-whinnies-osa/*.WAV'))
        dataset = add_n_files_to_dataset(n=osa_n_pos, folder='osa-1-negatives', dataset=dataset, example_type=0, sr=sr, noise=noise, noise_samples=noise_samples)   
        # Catappa - 50%/50% random negatives, hard-negative mined negatives
        cat_n_pos = len(glob.glob('../Data/clipped-whinnies-catappa/*.WAV'))
        split = int(round(cat_n_pos/2.0))
        # hnm:
        dataset = add_n_files_to_dataset(n=split, folder='catappa-hnm', dataset=dataset, example_type=0, sr=sr, noise=noise, noise_samples=noise_samples)
        # random:
        dataset = add_n_files_to_dataset(n=cat_n_pos-split, folder='catappa-2-negatives', dataset=dataset, example_type=0, sr=sr, noise=noise, noise_samples=noise_samples)
        # Shady:
        shady_n_pos = len(glob.glob('../Data/clipped-whinnies-shady/*.WAV'))
        # hnm:
        dataset = add_n_files_to_dataset(n=shady_n_pos, folder='catappa-hnm', dataset=dataset, example_type=0, sr=sr, noise=noise, noise_samples=noise_samples)
        # no negatives from Corcovado, so remainder made up with non-monkey-noises:
        remainder = wavtools.num_examples(dataset,1) - wavtools.num_examples(dataset,0)
        dataset = add_n_files_to_dataset(n=remainder, folder='non-monkey-noises', dataset=dataset, example_type=0, sr=sr, noise=noise, noise_samples=noise_samples)
        
        return dataset

    def add_cropped_dataset(dataset, noise=False, noise_samples=False):
        
        print "Adding basic dataset..."
        dataset = add_basic_dataset(dataset,noise=False,noise_samples=False)
        print "Adding augmented cropped data..."
        print "...augmented positives"
        dataset = add_n_files_to_dataset(n=-1, folder='aug-timeshifted', dataset=dataset, example_type=1, sr=sr)
        print "...augmented negatives"
        remainder = wavtools.num_examples(dataset,1) - wavtools.num_examples(dataset,0)
        split = int(round(remainder/2.0))
        dataset = add_n_files_to_dataset(n=split, folder='catappa-2-hnm', dataset=dataset, example_type=0, sr=sr)
        dataset = add_n_files_to_dataset(n=remainder-split, folder='catappa-2-negatives', dataset=dataset, example_type=0, sr=sr)

        return dataset

    # run-type optional processing methods
    if run_type == 'without-preprocessing':
        dataset = []
        dataset = add_basic_dataset(dataset,noise=False,noise_samples=False)
    elif run_type == 'standardised':
        dataset = []
        dataset = add_basic_dataset(dataset,noise=False,noise_samples=False)
        standardise_dataset(dataset) 
    elif run_type == 'denoised':
        dataset = []
        dataset = add_basic_dataset(dataset,noise=False,noise_samples=False)
        denoise_dataset(dataset)
    elif run_type == 'denoised/standardised':
        dataset = []
        dataset = add_basic_dataset(dataset,noise=False,noise_samples=False)
        standardise_dataset(dataset) 
        denoise_dataset(dataset)    
    elif run_type == 'crop_augmented':
        dataset = []
        dataset = add_cropped_dataset(dataset,noise=False,noise_samples=False)
    elif run_type == 'all_aug':
        dataset = []
        dataset = add_cropped_dataset(dataset,noise=False,noise_samples=False)
        print "\nAdding noise augmented data..."
        dataset = add_cropped_dataset(dataset,noise=True,noise_samples=False)
        print "\nAdding noise sample augmented data..."
        dataset = add_cropped_dataset(dataset,noise=False,noise_samples=True)

    print "\nNumber of samples in dataset: " + \
    str(wavtools.num_examples(dataset,0)) + " negative, " + \
    str(wavtools.num_examples(dataset,1)) + " positive"

    return dataset

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

def model_train(dataset, name, train_perc, num_epochs, batch_size):
    """
    Trains and saves simple keras model. 
    """
    try:
        random.shuffle(dataset)
    except NameError:
        print 'non-existent dataset name provided. check dataset exists and retry'
        return 

    # use provided training percentage to give num. training samples
    n_train_samples = int(round(len(dataset)*train_perc))
    train = dataset[:n_train_samples]
    # tests on remaining % of total
    test = dataset[n_train_samples:]    

    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)

    # reshape for CNN input
    x_train = np.array([x.reshape( (128, 282, 1) ) for x in x_train])
    x_test = np.array([x.reshape( (128, 282, 1) ) for x in x_test])

    # # one-hot encoding for classes
    # y_train = np.array(keras.utils.to_categorical(y_train, 2))
    # y_test = np.array(keras.utils.to_categorical(y_test, 2))

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder.fit(y_test)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    model = Sequential()
    input_shape=(128, 282, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('elu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.8364280725819752)) # from hyperas

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.9532739136720357)) # from hyperas

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=['accuracy', precision, recall, f1]) 

    history = model.fit( # was 'model.fit('
        x=x_train, 
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data= (x_test, y_test))
        # callbacks=[metrics])

    score = model.evaluate(
        x=x_test,
        y=y_test)
    
    # y_pred = model.predict(x_test)

    # list all data in history
    # print history.history.keys() 
   
    # custom function to create plots of training behaviour
    def training_behaviour_plot(metric):
        """
        Produces and saves plot of given metric for training
        and test datasets over duration of training time

        e.g. training_behaviour_plot('recall')
        """
        # check/make savepath 
        plot_save_path = '../Results/CNN-learning-behaviour-plots/'+'model_'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        # compile and save plot for given metric  
        file_name = plot_save_path+'/'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)+'_'+metric+'.png'
        # remove plot if already exists, to allow to be overwritten:
        if os.path.isfile(file_name):
            os.remove(file_name)
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title("Change in metric '"+metric+ "' over total training time")
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(file_name)
        plt.gcf().clear()
        print('saved '+metric+' plot to ../Results/CNN-learning-behaviour-plots/')

    print('\n')
    training_behaviour_plot('acc')
    training_behaviour_plot('loss')
    training_behaviour_plot('recall')
    training_behaviour_plot('precision')
    training_behaviour_plot('f1')

    # def auc_roc_plot(X_test, y_test):
    #     # generate ROC
    #     y_pred = model.predict(X_test).ravel()
    #     fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #     # generate AUC
    #     auc_val = auc(fpr, tpr)
    #     # plot ROC
    #     plot_save_path = '../Results/CNN-learning-behaviour-plots/'+'model_'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)
    #     file_name = plot_save_path+'/'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)+'_ROC.png'
    #     plt.figure(1)
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.plot(fpr, tpr, label='area under curve = {:.3f})'.format(auc_val))
    #     plt.xlabel('False positive rate')
    #     plt.ylabel('True positive rate')
    #     plt.title('ROC curve - '+name+'_e'+str(num_epochs)+'_b'+str(batch_size))
    #     plt.legend(loc='lower right')
    #     plt.savefig(file_name)
    #     plt.gcf().clear()
    #     print('saved ROC plot to ../Results/CNN-learning-behaviour-plots/')

    #     return auc_val

    # return auc roc value and save roc plot to results folder
    # auc_val = auc_roc_plot(X_test, y_test)

    print('\nLearning rate:', str(K.eval(model.optimizer.lr)))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # print('AUC ROC:', auc_val)
    # print('AUC pr:', history.history[auc_pr])

    # metrics.get_data()

    # serialise model to JSON
    dataset_name = name
    save_path = '/home/dgabutler/Work/CMEEProject/Models/'+dataset_name+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_json = model.to_json()
    with open(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialise weights to HDF5
    model.save_weights(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.h5')
    print '\nsaved model '+dataset_name+'/'+'e'+str(num_epochs)+'_b'+str(batch_size)+' to disk' 

# D_original = compile_dataset('without-preprocessing', 48000)
# D_standard = compile_dataset('standardised', 48000)
# D_denoised = compile_dataset('denoised/standardised', 48000)
# D_crop_aug = compile_dataset('crop_augmented', 48000)
# D_all_aug = compile_dataset('all_aug', 48000)

# dataset = D_all_aug
# name = 'D_all_aug'
# train_perc = 0.9
# num_epochs = 50
# batch_size = 32

# model_train(dataset, name, train_perc, num_epochs, batch_size)



def load_keras_model(dataset, model_name):
    """
    Loads pretrained model from disk for a given dataset type.
    """    
    folder_path = '../Models/'
    model_path = folder_path + dataset + '/' + model_name + '_model.json'
    try:
        json_file = open(model_path, 'r')
    except IOError:
        print('\nError: no model exists for that dataset name at the provided path: '+model_path+'\n\nCheck and try again')
        return 
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into model
    loaded_model.load_weights(folder_path + dataset + '/' + model_name + '_model.h5')
    print("\nLoaded model from disk")

    return loaded_model 

# model = load_keras_model('D_all_aug','e30_b32')


def search_file_for_monkeys(file_name, threshold_confidence, wav_folder, model, tidy=True, full_verbose=True, hnm=False, summary_file=False):
    """
    Splits 60-second file into 3-second clips. Runs each through
    detector. If activation surpasses confidence threshold, clip
    is separated.
    If hard-negative mining functionality selected, function
    takes combination of labelled praat file and 60-second wave file,
    runs detector on 3-second clips, and seperates any clips that 
    the detector incorrectly identifies as being positives.
    These clips are then able to be fed in as negative examples, to
    improve the discriminatory capability of the network 

    Example call: search_file_for_monkeys('5A3AD7A6', 60, '/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/shady-lane/')
    """
    audio_folder = wav_folder
    # isolate folder name from path:
    p = pathlib.Path(wav_folder)
    isolated_folder_name = p.parts[2:][-1]
    wav = audio_folder+file_name+'.WAV'
    try:
        wavfile = AudioSegment.from_wav(wav)
    except OSError:
        print "\nerror: audio file",os.path.basename(wav),"at path", os.path.dirname(wav), "cannot be loaded - probably improperly recorded"
        return 
    clip_length_ms = 3000
    clips = make_chunks(wavfile, clip_length_ms)

    print "\n-- processing file " + file_name +'.WAV'

    # if hard-negative mining, test for presence of praat file early for efficiency:
    if hnm:
        praat_file_path = '/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file_name+'.TextGrid'
        try:
            labelled_starts = wavtools.whinny_starttimes_from_praatfile(praat_file_path)[1]

        except IOError:
            print 'error: no praat file named',os.path.basename(praat_file_path),'at path', os.path.dirname(praat_file_path)
            return

    clip_dir = wav_folder+'clips-temp/'
    # delete temporary clips directory if interuption to previous
    # function call failed to remove it 
    if os.path.exists(clip_dir) and os.path.isdir(clip_dir):
        rmtree(clip_dir)
    # create temporary clips directory 
    os.makedirs(clip_dir) 

    # Export all inviduals clips as wav files
    # print 'clipping 60 second audio file into 3 second snippets to test...\n'
    for clipping_idx, clip in enumerate(clips):
        clip_name = "clip{0:02}.wav".format(clipping_idx+1)
        clip.export(clip_dir+clip_name, format="wav")

    D_test = [] 

    clipped_wavs = glob.glob(clip_dir+'clip*')
    clipped_wavs.sort(key=lambda f: int(filter(str.isdigit, f)))

    for clip in clipped_wavs:
        y, sr = librosa.load(clip, sr=None, duration=3.00)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 282): continue
        D_test.append(ps)

    # D_test = wavtools.denoise_dataset(D_test)

    call_count = 0
    hnm_counter = 0

    # reshape to be correct dimension for CNN input
    # NB. dimensions are: num.samples, num.melbins, num.timeslices, num.featmaps 
    # print "...checking clips for monkeys..."
    for idx, clip in enumerate(D_test):
        D_test[idx] = clip.reshape(1,128,282,1)
        predicted = model.predict(D_test[idx])

        # if NEGATIVE:
        if predicted[0][0] <= (threshold_confidence/100.0): ########## THIS IS SECTION THAT CHANGED BETWEEN 1 node/2 node:
            continue                                        # WAS: if predicted[0][1] <= (threshold_confidence/100.0)
                                                            # furthermore 3 changes (predicted[0][1] -> ..cted[0][0]) below 

        else:
        # if POSITIVE
            call_count+=1
            lower_clip_bound = (3*(idx+1))-3
            upper_clip_bound = 3*(idx+1)
            # i.e. clip 3 would be 6-9 seconds into original 60-sec file
            approx_position = str(lower_clip_bound)+'-'+str(upper_clip_bound)

            # regular detector behaviour - not hard negative mining
            if not hnm:
                # suspected positives moved to folder in Results, files renamed 'filename_numcallinfile_confidence.WAV'
                # results_dir = '/media/dgabutler/My Passport/Audio/detected-positives/'+isolated_folder_name+'-results'
                results_dir = '/home/dgabutler/Work/CMEEProject/Results/detected-positives/'+isolated_folder_name+'-results'

                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                copyfile(clipped_wavs[idx], results_dir+'/'+file_name+'_'+str(call_count)+'_'+approx_position+'_'+str(int(round(predicted[0][0]*100)))+'.WAV')

                # making summary file 
                if summary_file:
                    summary_file_name = '/home/dgabutler/Work/CMEEProject/Results/'+isolated_folder_name+'-results-summary.csv'
                    # obtain datetime from file name if possible 
                    try:
                        datetime_of_recording = wavtools.filename_to_localdatetime(file_name)
                        date_of_recording = datetime_of_recording.strftime("%d/%m/%Y")
                        time_of_recording = datetime_of_recording.strftime("%X")
                    # if not possible due to unusual file name, 
                    # assign 'na' value to date time 
                    except ValueError:
                        date_of_recording = 'NA'
                        time_of_recording = 'NA' 
                    
                    # values to be entered in row of summary file:
                    column_headings = ['file name', 'approx. position in recording (secs)', 'time of recording', 'date of recording', 'confidence']
                    csv_row = [file_name, approx_position, time_of_recording, date_of_recording, str(int(round(predicted[0][0]*100)))+'%']
                        
                    # make summary file if it doesn't already exist
                    summary_file_path = pathlib.Path(summary_file_name)
                    if not summary_file_path.is_file():
                        with open(summary_file_name, 'wb') as csvfile:
                            filewriter = csv.writer(csvfile, delimiter=',')
                            filewriter.writerow(column_headings)
                            filewriter.writerow(csv_row)
                    
                    # if summary file exists, *append* row to it
                    else:
                        with open(summary_file_name, 'a') as csvfile:
                            filewriter = csv.writer(csvfile, delimiter=',')
                            filewriter.writerow(csv_row)
            else:
            # if hard-negative mining for false positives to enhance training:
                labelled_ends = wavtools.whinny_endtimes_from_praatfile(praat_file_path)[1]

                if not any(lower_clip_bound <= starts/1000.0 <= upper_clip_bound for starts in labelled_starts) \
                and not any(lower_clip_bound <= ends/1000.0 <= upper_clip_bound for ends in labelled_ends):                   
                    # i.e. if section has not been labelled as containing a call
                    # (therefore a false positive has been detected)
                    hnm_counter+=1
                    copyfile(clipped_wavs[idx], '/home/dgabutler/Work/CMEEProject/Data/mined-false-positives/'+file_name+'_'+str(hnm_counter)+'_'+approx_position+'_'+str(int(round(predicted[0][0]*100)))+'.WAV')
                else: continue     

        # if full_verbose:
        #     print 'clip number', '{0:02}'.format(idx+1), '- best guess -', best_guess

    # delete all created clips and temporary clip folder
    if tidy:
        rmtree(clip_dir)
        # empty recycling bin to prevent build-up of trashed clips
        subprocess.call(['rm -rf /home/dgabutler/.local/share/Trash/*'], shell=True)

    # print statements to terminal
    if full_verbose:
        if not hnm:
            print '\nfound', call_count, 'suspected call(s) that surpass %d%% confidence threshold in 60-second file %s.WAV' % (threshold_confidence, file_name)
        else:
            print '\nhard negative mining generated', hnm_counter, 'suspected false positive(s) from file', file_name, 'for further training of network'

def hard_negative_miner(wav_folder, threshold_confidence, model):

    # list all file names in folder
    wavs = glob.glob(wav_folder+'*.WAV')
    wavs = [os.path.splitext(x)[0] for x in wavs]
    wavs = [os.path.basename(x) for x in wavs]

    for wav in wavs:
        search_file_for_monkeys(wav, threshold_confidence=threshold_confidence, wav_folder=wav_folder, model=model, hnm=True)

def search_folder_for_monkeys(wav_folder, threshold_confidence, model):
    
    # list all file names in folder
    wavs = glob.glob(wav_folder+'*.WAV')
    wavs = [os.path.splitext(x)[0] for x in wavs]
    wavs = [os.path.basename(x) for x in wavs]

    # require user input if code is suspected to take long to run
    predicted_run_time = len(wavs)*1.553
    if len(wavs) > 30:
        confirmation = raw_input("\nwarning: this code will take approximately " + str(round(predicted_run_time/60, 3)) + " minutes to run. enter Y to proceed\n\n")
        if confirmation != "Y":
            print '\nerror: function terminating as permission not received'
            return 
    tic = time.time()

    for wav in wavs:
        search_file_for_monkeys(wav, threshold_confidence=threshold_confidence, wav_folder=wav_folder, model=model, full_verbose=False, summary_file=True)

    toc = time.time()
    print '\nsystem took', round((toc-tic)/60, 3), 'mins to process', len(wavs), 'files\n\nfor a summary of results, see the csv file created in Results folder\n' 

# search_folder_for_monkeys('../Data/unclipped-whinnies-osa/',60,model)