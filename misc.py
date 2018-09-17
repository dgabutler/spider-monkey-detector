# MISC USEFUL CODE SNIPPETS. NOT USED IN WORKING VERSION,
# BUT MAY WISH TO COME BACK TO

# from wavtools.py

def other_starttimes_from_praatfile(praat_file):
    """
    Extracts 'other' start times (in milliseconds) from praat text file
    Output is tuple containing .wav file name and then all times (in ms)
    """
    start_times = []    # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()
    
    line_with_wavname = praat_contents[10]
    result = re.search('"(.*)"', line_with_wavname)
    wav_name = result.group(1)

    for idx, line in enumerate(praat_contents): 
        if "_Other" in line and "intervals" in praat_contents[idx+1]:
            time_line = praat_contents[idx-2]
            start_times.extend(re.findall("\d+\.\d+", time_line))

    start_times = map(float, start_times) # converts time to number, not 
                                          # character string
    
    # Comment line below if time in seconds is wanted:
    start_times = [times * 1000 for times in start_times]

    return wav_name, start_times

def other_endtimes_from_praatfile(praat_file):
    """
    Extracts 'other' end times (in milliseconds) from praat text file
    Output is tuple containing .wav file name and then all times in ms
    """
    end_times = []    # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()
    
    line_with_wavname = praat_contents[10]
    result = re.search('"(.*)"', line_with_wavname)
    wav_name = result.group(1)

    for idx, line in enumerate(praat_contents): 
        if "_Other" in line and "intervals" in praat_contents[idx+1]:
            time_line = praat_contents[idx-1]
            end_times.extend(re.findall("\d+\.\d+", time_line))
    
    end_times = map(float, end_times) # converts time to number, not 
                                      # character string
    
    # Comment line below if time in seconds is wanted:
    end_times = [times * 1000 for times in end_times]
    
    return wav_name, end_times

def do_augmentation(mag_spec, sr, roll=True, noise=True, noise_samples=True):
    """
    Fundamentally similar to augmentation function in git repo
    of Kahl et al. 2017
    For a given file, will augment magnitude spectrogram using one 
    of any combination of the following methods:
    - vertical pitch roll 
    - add an amount of salt-and-pepper Gaussian noise
    - blend with a file containing a non-spider-monkey-noise
    e.g. howler calling, cockerel crowing, any number of diff.
    birds, 
    and return mel-frequency spectrogram 
    """
    # alter prob. of augmentation depending on combination chosen
    if roll and not noise and not noise_samples:
        roll_prob = 1.0
    elif noise and not noise_samples and not roll:
        noise_prob = 1.0
    elif noise_samples and not noise and not roll:
        noise_samples_prob = 1.0
    else:
        roll_prob = 0.5
        noise_prob = 0.5
        noise_samples_prob = 0.2
    
    AUG = {#'type':[probability, value]
        'roll':[roll_prob, (0.0, 0.05)], # prob. was 0.5
        'noise':[noise_prob, 0.01], # prob. was 0.1
        'noise_samples':[noise_samples_prob, 1.0], # prob. was 0.1
        }

    RANDOM = np.random.RandomState(100)

    # wrap shift (roll up/down) NB. left/right commented out
    if 'roll' and RANDOM.choice([True, False], p=[AUG['roll'][0], 1 - AUG['roll'][0]]):
        mag_spec = np.roll(mag_spec, int(mag_spec.shape[0] * (RANDOM.uniform(-AUG['roll'][1][1], AUG['roll'][1][1]))), axis=0)
        # img = np.roll(img, int(img.shape[1] * (RANDOM.uniform(-AUG['roll'][1][0], AUG['roll'][1][0]))), axis=1)

    # gaussian noise
    if 'noise' and RANDOM.choice([True, False], p=[AUG['noise'][0], 1 - AUG['noise'][0]]):
        mag_spec = mag_spec + RANDOM.normal(0.0, RANDOM.uniform(0, AUG['noise'][1]**0.5), mag_spec.shape)
        mag_spec = np.clip(mag_spec, 0.0, 1.0)

    # load noise samples
    NOISE = shuffle(glob.glob('../Data/non-monkey-noises/*.WAV'))

    # add noise samples
    if 'noise_samples' and RANDOM.choice([True, False], p=[AUG['noise_samples'][0], 1 - AUG['noise_samples'][0]]):
        mag_spec = mag_spec + load_mag_spec(NOISE[RANDOM.choice(range(0, len(NOISE)))],sr=sr) * AUG['noise_samples'][1]
        mag_spec -= mag_spec.min(axis=None)
        mag_spec /= mag_spec.max(axis=None)
    
    aug_mel_spec = librosa.feature.melspectrogram(S=mag_spec, sr=sr)
    return aug_mel_spec 

def load_mag_spec(path, sr, winlen=0.025, winstep=0.01, NFFT=2048, denoise=True, normalize=True):
    """
    From https://stackoverflow.com/questions/42330830/how-should-audio-be-pre-processed-for-classification
    Change: winstep to alter num. timesteps, 
            NFFT to alter num. frequency bins
    """
    #open wav file
    (sig,rate) = librosa.load(path, sr=sr, duration=3.00)

    #get frames
    winfunc=lambda x:np.ones((x,))
    frames = psf.sigproc.framesig(sig, winlen*rate, winstep*rate, winfunc)

    #magnitude Spectrogram
    magspec = np.rot90(psf.sigproc.magspec(frames, NFFT))

    #noise reduction (mean substract)
    if denoise:
        magspec -= magspec.mean(axis=0)

    #normalize values between 0 and 1
    if normalize:
        magspec -= magspec.min(axis=0)
        magspec /= magspec.max(axis=0)

    return magspec

def add_files_to_dataset(folder, dataset, example_type, sr):
    """
    Takes all wav files from given folder name (minus slashes on 
    either side) and adds to the dataset name provided.
    Example type = 0 if negative, 1 if positive.
    sr (sampling rate) must be 44100 or 48000. 
    """
    data_folder_path = '../Data/'
    files = glob.glob(data_folder_path+folder+'/*.WAV')
    for wav in files:
        y, sr = librosa.load(wav, sr=sr, duration=3.00)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if sr == 48000:
            if ps.shape != (128, 282): continue
        elif sr == 44100:
            if ps.shape != (128, 259): continue
        else:
            return("error: sampling rate must be 48000 or 44100") 
        dataset.append((ps, example_type))

def custom_add_files_to_dataset(folder, dataset, example_type, sr, denoise=False, normalize=False, augment=False):
    """
    Takes all wav files from given folder name (minus slashes on 
    either side) and adds to the dataset name provided.
    Example type = 0 if negative, 1 if positive.
    sr (sampling rate) must be 44100 or 48000. 
    """
    data_folder_path = '../Data/'
    append_index = len(dataset)
    files = glob.glob(data_folder_path+folder+'/*.WAV')
    appended_region = np.zeros((len(files),2), dtype=object)
    for wav in files:
        mag_spec = load_mag_spec(wav, sr, denoise=denoise, normalize=normalize)
        if augment:
            mel_spec = do_augmentation(mag_spec, sr, noise=True, noise_samples=True, roll=True)
        else: 
            mel_spec = librosa.feature.melspectrogram(S=mag_spec, sr=sr)
        if mel_spec.shape != (128, 299): continue

        dataset[append_index] = [mel_spec, example_type]
        append_index += 1

# (potentially not useful):

def augment_file_blend(file_name):
    """
    Take file from aug time shift folder. use no overlap tho, i.e. just have file move in time 
    Blend it with random noise 
    Return resulting wav file 
    """
    noise_files = glob.glob('/home/dgabutler/Work/CMEEProject/Data/non-monkey-noises/*.WAV')
    signal = AudioSegment.from_wav(file_name)
    noise_file = random.choice(noise_files)
    noise = AudioSegment.from_wav(noise_file)

    # naming
    signal_name = os.path.basename(os.path.splitext(file_name)[0])
    noise_name = os.path.basename(os.path.splitext(noise_file)[0])
    aug_name = signal_name + '_aug_' + noise_name 

    # mix files together
    combined = signal.overlay(noise)
    combined.export("/home/dgabutler/Work/CMEEProject/Data/aug-blended/"+aug_name+'.WAV', format='wav')

def augment_file_vertical_roll(spectrogram):
    """Fundamentally similar to roll function in git repo
    of Kahl et al. 2017
    For a given spectrogram, will randomly pitch shift it 
    up or down by small amount.
    E.g. np.roll(spect, -1, axis=0) would move everything
    up by 1 in the frequency bins dimension
    """
    RANDOM = np.random.RandomState(100)

    IM_AUGMENTATION = {#'type':[probability, value]
                    'roll':[0.5, (0.0, 0.05)], 
                    'noise':[0.1, 0.01],
                    'noise_samples':[0.1, 1.0],
                    }
    AUG = IM_AUGMENTATION

    spectrogram = np.roll(spectrogram, int(spectrogram.shape[0] * (RANDOM.uniform(-AUG['roll'][1][1], AUG['roll'][1][1]))), axis=0)
    return spectrogram 

def augment_gaussian_noise(spectrogram):
        RANDOM = np.random.RandomState(100)
        spectrogram += RANDOM.normal(0.0, RANDOM.uniform(0, 0.1), spectrogram.shape)
        spectrogram = np.clip(spectrogram, 0.0, 1.0)
        return spectrogram

# from keras_classifier

def train_simple_keras(dataset, name, train_perc, num_epochs, batch_size):
    """
    Trains and saves simple keras model. 
    """
    try:
        random.shuffle(dataset)
    except NameError:
        print('non-existent dataset name provided. check dataset exists and retry')
        return 

    # use provided training percentage to give num. training samples
    n_train_samples = int(round(len(dataset)*train_perc))
    train = dataset[:n_train_samples]
    # tests on remaining % of total
    test = dataset[n_train_samples:]    

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # reshape for CNN input
    X_train = np.array([x.reshape( (128, 299, 1) ) for x in X_train])
    X_test = np.array([x.reshape( (128, 299, 1) ) for x in X_test])

    # one-hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, 2))
    y_test = np.array(keras.utils.to_categorical(y_test, 2))

    model = Sequential()
    input_shape=(128, 299, 1)

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

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # following two lines are experimenting with diff. metric method
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=['accuracy', precision, recall]) 

    history = model.fit( # was 'model.fit('
        x=X_train, 
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data= (X_test, y_test))

    score = model.evaluate(
        x=X_test,
        y=y_test)

    # # list all data in history
    # print(history.history.keys()) # added 

    # custom function to create plots of training behaviour
    def training_behaviour_plot(metric):
        """produces and saves plot of given metric for training
        and test datasets over duration of training time

        e.g. training_behaviour_plot('recall')
        """
        # check/make savepath 
        plot_save_path = '../Results/CNN-learning-behaviour-plots/two-node-output/'+'model_'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        # compile and save plot for given metric  
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title('model '+name+''+'_e'+str(num_epochs)+'_b'+str(batch_size)+' '+metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        file_name = plot_save_path+'/'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)+'_'+metric+'.png'
        # remove plot if already exists, to allow to be overwritten:
        if os.path.isfile(file_name):
            os.remove(file_name)
        plt.savefig(file_name)
        print('saved '+metric+' plot to '+plot_save_path)
        plt.gcf().clear()

    training_behaviour_plot('acc')
    training_behaviour_plot('loss')
    training_behaviour_plot('recall')
    training_behaviour_plot('precision')

    print('\nlearning rate:', str(K.eval(model.optimizer.lr)))
    print('test loss:', score[0])
    print('test accuracy:', score[1])

    print('\nlearning rate:', str(K.eval(model.optimizer.lr)))

    print('test loss:', score[0])
    print('test accuracy:', score[1])

    # serialise model to JSON
    dataset_name = name
    save_path = '/home/dgabutler/Work/CMEEProject/Models/two-node-output/'+dataset_name+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_json = model.to_json()
    with open(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialise weights to HDF5
    model.save_weights(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.h5')
    print('\nsaved model '+dataset_name+'/'+'e'+str(num_epochs)+'_b'+str(batch_size)+' to disk')

def train_simple_keras_ONE_NODE_OUTPUT(dataset, name, train_perc, num_epochs, batch_size, sr):
    """
    Adapted train-and-save function, with
    SINGLE SIGMOID OUTPUT LAYER replacing the two nodes
    - decision taken following advice of Harry Berg, w/
    guidance from https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
    """
    try:
        random.shuffle(dataset)
    except NameError:
        print('Non-existent dataset name provided. Check dataset exists and retry')
        return 

    # increased sampling rate gives increased number of time-slices per clip
    # this affects CNN input size, and time_dimension used as proxy to ensure 
    # all clips tested are of the same length (if not, they are not added to test dataframe)
    if sr == 44100:
        time_dimension = 299
    elif sr == 48000:
        time_dimension = 299
    else:
        return("error: sampling rate must be 48000 or 44100") 

    # use provided training percentage to give num. training samples
    n_train_samples = int(round(len(dataset)*train_perc))
    train = dataset[:n_train_samples]
    # tests on remaining % of total
    test = dataset[n_train_samples:]    

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # reshape for CNN input
    X_train = np.array([x.reshape((128, time_dimension, 1)) for x in X_train])
    X_test = np.array([x.reshape((128, time_dimension, 1)) for x in X_test])

    # ALTERED ENCODING SECTION #######################################
    # previously was:
    # # one-hot encoding for classes
    # y_train = np.array(keras.utils.to_categorical(y_train, 2))
    # y_test = np.array(keras.utils.to_categorical(y_test, 2))

    # changed to...
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder.fit(y_test)
    encoded_y_train = encoder.transform(y_train)
    encoded_y_test = encoder.transform(y_test)
    ##################################################################

    model = Sequential()
    input_shape=(128, time_dimension, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5)) # can optimise

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5)) # can optimise

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # following two lines are experimenting with diff. metric method
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        # metrics=['accuracy', precision_threshold(0.5), recall_threshold(0.5)]) 
        metrics=['accuracy', precision, recall, auc_pr])

    history = model.fit( 
        x=X_train, 
        y=encoded_y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data= (X_test, encoded_y_test))

    score = model.evaluate(
        x=X_test,
        y=encoded_y_test)

    # # list all data in history
    # print(history.history.keys()) # added 
    print('\n')

    # custom function to create plots of training behaviour
    def training_behaviour_plot(metric):
        """
        Produces and saves plot of given metric for training
        and test datasets over duration of training time

        e.g. training_behaviour_plot('recall')
        """
        # check/make savepath 
        plot_save_path = '../Results/CNN-learning-behaviour-plots/one-node-output/'+'model_'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        # compile and save plot for given metric  
        file_name = plot_save_path+'/'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)+'_'+metric+'.png'
        # remove plot if already exists, to allow to be overwritten:
        if os.path.isfile(file_name):
            os.remove(file_name)
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title('model '+name+''+'_e'+str(num_epochs)+'_b'+str(batch_size)+' '+metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(file_name)
        plt.gcf().clear()
        print('saved '+metric+' plot to ../Results/CNN-learning-behaviour-plots/')

    training_behaviour_plot('acc')
    training_behaviour_plot('loss')
    training_behaviour_plot('recall')
    training_behaviour_plot('precision')

    def auc_roc_plot(X_test, y_test):
        # generate ROC
        y_pred = model.predict(X_test).ravel()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # generate AUC
        auc_val = auc(fpr, tpr)
        # plot ROC
        plot_save_path = '../Results/CNN-learning-behaviour-plots/one-node-output/'+'model_'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)
        file_name = plot_save_path+'/'+name+'_e'+str(num_epochs)+'_b'+str(batch_size)+'_ROC.png'
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='area under curve = {:.3f})'.format(auc_val))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve - '+name+'_e'+str(num_epochs)+'_b'+str(batch_size))
        plt.legend(loc='lower right')
        plt.savefig(file_name)
        plt.gcf().clear()
        print('saved ROC plot to ../Results/CNN-learning-behaviour-plots/')

        return auc_val

    # return auc roc value and save roc plot to results folder
    auc_val = auc_roc_plot(X_test, y_test)

    print('\nLearning rate:', str(K.eval(model.optimizer.lr)))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('AUC ROC:', auc_val)
    print('AUC pr:', history.history[auc_pr])

    # serialise model to JSON
    dataset_name = name
    save_path = '/home/dgabutler/Work/CMEEProject/Models/one-node-output/'+dataset_name+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_json = model.to_json()
    with open(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialise weights to HDF5
    model.save_weights(save_path+'e'+str(num_epochs)+'_b'+str(batch_size)+'_model.h5')
    print('\nSaved model '+dataset_name+'/'+'e'+str(num_epochs)+'_b'+str(batch_size)+' to disk')

def load_keras_model(dataset, model_name):
    """
    Loads pretrained model from disk for a given dataset type.
    """    
    folder_path = '../Models/one-node-output/'
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

def search_file_for_monkeys_TWO_NODE_OUTPUT(file_name, threshold_confidence, wav_folder, model, denoise=True, standardise=True, tidy=True, full_verbose=True, hnm=False, summary_file=False):
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

    Options include denoising and standardising of input audio files.

    This function is the old version of search_file_for_monkeys, when 
    CNN terminated in two nodes in output layer.
    Updated version, with only a sigmoid-activated single node, is 
    the function that search_folder_ & search_file_list use as default.

    This version is no longer being updated with latest additions, e.g.:

    - does not have sampling rate argument

    Example call: 
    search_file_for_monkeys_TWO_NODE_OUTPUT('5A3BE710', 60, '/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/shady-lane/', loaded_model)
    """

    # isolate folder name from path:
    p = pathlib.Path(wav_folder)
    isolated_folder_name = p.parts[2:][-1]
    wav = wav_folder+file_name+'.WAV'
    # checks: does audio file exist and can it be read
    if not os.path.isfile(wav):
        print("\nerror: no audio file named",os.path.basename(wav),"at path", os.path.dirname(wav))
        return 
    try:
        wavfile = AudioSegment.from_wav(wav)
    except OSError:
        print("\nerror: audio file",os.path.basename(wav),"at path",os.path.dirname(wav),"exists but cannot be loaded (probably improperly recorded)")
        return 
    clip_length_ms = 3000
    clips = make_chunks(wavfile, clip_length_ms)

    print("\n-- processing file " + file_name +'.WAV')

    # if hard-negative mining, test for presence of praat file early for efficiency:
    if hnm:
        praat_file_path = '/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file_name+'.TextGrid'
        try:
            labelled_starts = wavtools.whinny_starttimes_from_praatfile(praat_file_path)[1]

        except IOError:
            print('error: no praat file named',os.path.basename(praat_file_path),'at path', os.path.dirname(praat_file_path))
            return

    clip_dir = wav_folder+'clips-temp/'

    # delete temporary clips directory if interuption to previous
    # function call failed to remove it 
    if os.path.exists(clip_dir) and os.path.isdir(clip_dir):
        rmtree(clip_dir)
    # create temporary clips directory 
    os.makedirs(clip_dir) 
    # export all inviduals clips as wav files
    # print('clipping 60 second audio file into 3 second snippets to test...\n')
    for clipping_idx, clip in enumerate(clips):
        clip_name = "clip{0:02}.wav".format(clipping_idx+1)
        clip.export(clip_dir+clip_name, format="wav")

    D_test = [] 

    clipped_wavs = glob.glob(clip_dir+'clip*')
    clipped_wavs = sorted(clipped_wavs, key=lambda item: (int(item.partition(' ')[0])
                                if item[0].isdigit() else float('inf'), item))

    # add each 3-second clip to dataframe for testing
    for clip in clipped_wavs:
        y, sr = librosa.load(clip, sr=None, duration=3.00)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 282): continue
        D_test.append(ps)

    # conditions for modifying file:
    if denoise == True:
        D_test = wavtools.denoise_dataset(D_test)
    if standardise == True:
        D_test = wavtools.standardise_inputs(D_test)

    # counters for naming of files
    call_count = 0
    hnm_counter = 0

    # reshape to be correct dimension for CNN input
    # NB. dimensions are: num.samples, num.melbins, num.timeslices, num.featmaps 
    # print("...checking clips for monkeys...")
    for idx, clip in enumerate(D_test):
        D_test[idx] = clip.reshape(1,128,282,1)
        predicted = model.predict(D_test[idx])

        # if NEGATIVE:
        if predicted[0][1] < (threshold_confidence/100.0):
            continue

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
                copyfile(clipped_wavs[idx], results_dir+'/'+file_name+'_'+str(call_count)+'_'+approx_position+'_'+str(int(round(predicted[0][1]*100)))+'.WAV')

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
                    csv_row = [file_name, approx_position, time_of_recording, date_of_recording, str(int(round(predicted[0][1]*100)))+'%']
                        
                    # make summary file if it doesn't already exist
                    summary_file_path = pathlib.Path(summary_file_name)
                    if not summary_file_path.is_file():
                        with open(summary_file_name, 'w') as csvfile:
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
                    copyfile(clipped_wavs[idx], '/home/dgabutler/Work/CMEEProject/Data/mined-false-positives/'+file_name+'_'+str(hnm_counter)+'_'+approx_position+'_'+str(int(round(predicted[0][1]*100)))+'.WAV')
                else: continue     

        # if full_verbose:
        #     print('clip number', '{0:02}'.format(idx+1), '- best guess -', best_guess)

    # delete all created clips and temporary clip folder
    if tidy:
        rmtree(clip_dir)
        # empty recycling bin to prevent build-up of trashed clips
        subprocess.call(['rm -rf /home/dgabutler/.local/share/Trash/*'], shell=True)

    # print statements to terminal
    if full_verbose:
        if not hnm:
            print('\nfound', call_count, 'suspected call(s) that surpass %d%% confidence threshold in 60-second file %s.WAV' % (threshold_confidence, file_name))
        else:
            print('\nhard negative mining generated', hnm_counter, 'suspected false positive(s) from file', file_name, 'for further training of network')

def search_file_for_monkeys_ONE_NODE_OUTPUT(file_name, threshold_confidence, wav_folder, model, sr, denoise=False, standardise=False, tidy=True, full_verbose=True, hnm=False, summary_file=False):
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

    Example call: 
    search_file_for_monkeys_ONE_NODE_OUTPUT('5A3BE710', 60, '/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/', loaded_model)

    NB. use following for testing, but REMEMBER TO DELETE***********
    file_name = '5A3BE710'
    threshold_confidence = 70
    wav_folder = '/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/'
    model = loaded_model

    """

    # increased sampling rate gives increased number of time-slices per clip
    # this affects CNN input size, and time_dimension used as proxy to ensure 
    # all clips tested are of the same length (if not, they are not added to test dataframe)
    if sr == 44100:
        time_dimension = 299
    elif sr == 48000:
        time_dimension = 299
    else:
        return("Error: sampling rate must be 48000 or 44100") 

    # isolate folder name from path:
    p = pathlib.Path(wav_folder)
    isolated_folder_name = p.parts[2:][-1]
    wav = wav_folder+file_name+'.WAV'
    # checks: does audio file exist and can it be read
    if not os.path.isfile(wav):
        print("\nError: no audio file named",os.path.basename(wav),"at path", os.path.dirname(wav))
        return 
    try:
        wavfile = AudioSegment.from_wav(wav)
    except OSError:
        print("\nError: audio file",os.path.basename(wav),"at path",os.path.dirname(wav),"exists but cannot be loaded (probably improperly recorded)")
        return 
    clip_length_ms = 3000
    clips = make_chunks(wavfile, clip_length_ms)

    print("\n-- processing file " + file_name +'.WAV')

    # if hard-negative mining, test for presence of praat file early for efficiency:
    if hnm:
        praat_file_path = '../Data/praat-files/'+file_name+'.TextGrid'
        try:
            labelled_starts = wavtools.whinny_starttimes_from_praatfile(praat_file_path)[1]

        except IOError:
            print('Error: no praat file named',os.path.basename(praat_file_path),'at path', os.path.dirname(praat_file_path))
            return

    clip_dir = wav_folder+'clips-temp/'

    # delete temporary clips directory if interuption to previous
    # function call failed to remove it 
    if os.path.exists(clip_dir) and os.path.isdir(clip_dir):
        rmtree(clip_dir)
    # create temporary clips directory 
    os.makedirs(clip_dir) 
    # export all inviduals clips as wav files
    # print('clipping 60 second audio file into 3 second snippets to test...\n')
    for clipping_idx, clip in enumerate(clips):
        clip_name = "clip{0:02}.wav".format(clipping_idx+1)
        clip.export(clip_dir+clip_name, format="wav")

    clipped_wavs = glob.glob(clip_dir+'clip*')
    clipped_wavs = sorted(clipped_wavs, key=lambda item: (int(item.partition(' ')[0])
                                if item[0].isdigit() else float('inf'), item))

    # preallocate dataframe of correct length
    D_test = np.zeros(len(clipped_wavs), dtype=object)

    for data_idx, clip in enumerate(clipped_wavs):
        # y, sr = librosa.load(clip, sr=sr, duration=3.00)
        mag_spec = wavtools.load_mag_spec(clip, sr, denoise=denoise, normalize=False)
        ps = librosa.feature.melspectrogram(S=mag_spec, sr=sr)
        if ps.shape != (128, time_dimension): continue
        D_test[data_idx] = ps

    # # conditions for modifying file:
    # if denoise == True:
    #     D_test = wavtools.denoise_dataset(D_test)
    # if standardise == True:
    #     D_test = wavtools.standardise_inputs(D_test)

    # counters for informative naming of files
    call_count = 0
    hnm_counter = 0

    # reshape to be correct dimension for CNN input
    # NB. dimensions are: num.samples, num.melbins, num.timeslices, num.featmaps 
    # print("...checking clips for monkeys...")
    for idx, clip in enumerate(D_test):
        D_test[idx] = clip.reshape(1,128,time_dimension,1)
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
                        with open(summary_file_name, 'w') as csvfile:
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
        #     print('clip number', '{0:02}'.format(idx+1), '- best guess -', best_guess)

    # delete all created clips and temporary clip folder
    if tidy:
        rmtree(clip_dir)
        # empty recycling bin to prevent build-up of trashed clips
        subprocess.call(['rm -rf /home/dgabutler/.local/share/Trash/*'], shell=True)

    # print statements to terminal
    if full_verbose:
        if not hnm:
            print('\nfound', call_count, 'suspected call(s) that surpass %d%% confidence threshold in 60-second file %s.WAV' % (threshold_confidence, file_name))
        else:
            print('\nhard negative mining generated', hnm_counter, 'suspected false positive(s) from file', file_name, 'for further training of network')

def search_folder_for_monkeys(wav_folder, threshold_confidence, model, sr):
    
    # list all file names in folder
    wavs = glob.glob(wav_folder+'*.WAV')
    wavs = [os.path.splitext(x)[0] for x in wavs]
    wavs = [os.path.basename(x) for x in wavs]

    # require user input if code is suspected to take long to run
    predicted_run_time = len(wavs)*1.553
    if len(wavs) > 30:
        confirmation = input("\nWarning: this code will take approximately " + str(round(predicted_run_time/60, 3)) + " minutes to run. enter Y to proceed\n\n")
        if confirmation != "Y":
            print('\nError: function terminating as permission not received')
            return 
    tic = time.time()

    for wav in wavs:
        search_file_for_monkeys_ONE_NODE_OUTPUT(wav, threshold_confidence, wav_folder, model, sr, full_verbose=False, summary_file=True)

    toc = time.time()
    print('\nSystem took', round((toc-tic)/60, 3), 'mins to process', len(wavs), 'files\n\nfor a summary of results, see the csv file created in Results folder\n')

def search_file_list_for_monkeys(file_names_list, wav_folder, threshold_confidence, model):
    """
    Search a list of provided file names for positives at given confidence
    """
    for name in file_names_list:
        try:
            search_file_for_monkeys_ONE_NODE_OUTPUT(name, wav_folder=wav_folder, threshold_confidence=threshold_confidence, model=model, full_verbose=False)
        except IOError:
            print('error: no file named', name)
            continue 

def custom_add_files_to_dataset(folder, dataset, example_type, sr, denoise=False, normalize=False, augment=False):
    """
    Takes all wav files from given folder name (minus slashes on 
    either side) and adds to the dataset name provided.
    Can append to a dataset already containing values, or to a 
    dataset that has only been initialised. 
    Numpy ndarrays and preallocation used for speed.

    Example type = 0 if negative, 1 if positive.
    sr (sampling rate) must be 44100 or 48000. 
    """
    data_folder_path = '/home/dgabutler/Work/CMEEProject/Data/'
    append_index = len(dataset)
    files = glob.glob(data_folder_path+folder+'/*.WAV')
    appended_region = np.zeros((len(files),2), dtype=object)
    dataset = np.vstack([dataset, appended_region])
    if augment:
        noise = input("\nNoise: True or False - ")
        noise_samples = input("\nNoise Samples: True or False - ")
        roll = input("\nRoll: True or False - ")
    for wav in files:
        mag_spec = wavtools.load_mag_spec(wav, sr, denoise=denoise, normalize=normalize)
        if augment:
            mel_spec = wavtools.do_augmentation(mag_spec, sr, noise=noise, noise_samples=noise_samples, roll=roll)
        else: 
            mel_spec = librosa.feature.melspectrogram(S=mag_spec, sr=sr)
        if mel_spec.shape != (128, 299): continue

        dataset[append_index] = [mel_spec, example_type]
        append_index += 1
    return dataset 

def compile_dataset(run_type, sr):
    """
    Data providing function.
    Run type argument specifies which set of data to load, 
    e.g. augmented, denoised.
    Current options are:
    - 'without-preprocessing'
    - 'denoised'
    - 'denoised/crop-aug' - coming soon
    - 'denoised/noise-aug'
    - 'denoised/crop-aug/unbalanced'
    - 'denoised/noise-sample-noise-roll-aug' - coming soon
    Returns dataset for input into CNN training function
    """
    # run-type optional processing methods
    if run_type == 'without-preprocessing':
        dataset = np.array([], dtype=object).reshape(0,2)
        # positives
        dataset = custom_add_files_to_dataset(folder='clipped-whinnies', dataset=dataset, example_type=1, sr=sr, denoise=False, normalize=False, augment=False)
        # negatives
        dataset = custom_add_files_to_dataset(folder='clipped-negatives', dataset=dataset, example_type=0, sr=sr, denoise=False, normalize=False, augment=False)
    if run_type == 'denoised':
        dataset = np.array([], dtype=object).reshape(0,2)
        # positives
        dataset = custom_add_files_to_dataset(folder='clipped-whinnies', dataset=dataset, example_type=1, sr=sr, denoise=True, normalize=False, augment=False)
        # negatives
        dataset = custom_add_files_to_dataset(folder='clipped-negatives', dataset=dataset, example_type=0, sr=sr, denoise=True, normalize=False, augment=False)
    if run_type == 'denoised/noise-aug':
        dataset = np.array([], dtype=object).reshape(0,2)
        # positives
        dataset = custom_add_files_to_dataset(folder='clipped-whinnies', dataset=dataset, example_type=1, sr=sr, denoise=True, normalize=False, augment=False)
        # positives
        dataset = custom_add_files_to_dataset(folder='clipped-whinnies', dataset=dataset, example_type=1, sr=sr, denoise=True, normalize=False, augment=False)
        # augmentations
        print('Select only noise to be true')
        dataset = custom_add_files_to_dataset(folder='clipped-negatives', dataset=dataset, example_type=0, sr=sr, denoise=True, normalize=False, augment=True)
    if run_type == 'denoised/crop-aug/unbalanced':
        dataset = np.array([], dtype=object).reshape(0,2)
        # positives
        dataset = custom_add_files_to_dataset(folder='clipped-whinnies', dataset=dataset, example_type=1, sr=sr, denoise=True, normalize=False, augment=False)
        # crop aug positives
        dataset = custom_add_files_to_dataset(folder='aug-timeshifted', dataset=dataset, example_type=1, sr=sr, denoise=True, normalize=False, augment=False)
        # balanced negatives
        dataset = custom_add_files_to_dataset(folder='clipped-negatives', dataset=dataset, example_type=0, sr=sr, denoise=True, normalize=False, augment=False)
        # unbalanced negatives
        dataset = custom_add_files_to_dataset(folder='catappa-2-confirmed-negatives', dataset=dataset, example_type=0, sr=sr, denoise=True, normalize=False, augment=False)
    
    print("\nNumber of samples in dataset: " + \
    str(wavtools.num_examples(dataset,0)) + " negative, " + \
    str(wavtools.num_examples(dataset,1)) + " positive")

    return dataset
