# Functions for general processing of .wav files 
# e.g. clipping using label files, preprocessing (denoising/normalizing)
# augmenting, and adding to datasets for training of CNNs 

# NB. few relative paths in script

from    datetime import datetime # used by filename_to_localdatetime
from    pytz import timezone # used by filename_to_localdatetime
import  pytz
import  re
import  os
from    pydub import AudioSegment  # used by: aug_time_shift, aug_file_blend
from    pydub.utils import make_chunks
import  os
import  numpy as np 
import  random # used by: aug_time_shift, aug_file_blend              
import  glob # used by: aug_file_blend       
import  librosa
import  librosa.display   
from    sklearn.utils import shuffle # used by do_augmentation
import  python_speech_features as psf # used by load_mag_spec


#######################
### WRANGLING FUNCTIONS

# useful:

def filename_to_localdatetime(filename):
    """
    Extracts datetime of recording in Costa Rica time from hexadecimal file name.
    Example call: filename_to_localdatetime('5A3AD5B6')
    """  
    time_stamp = int(filename, 16)
    naive_utc_dt = datetime.fromtimestamp(time_stamp)
    aware_utc_dt = naive_utc_dt.replace(tzinfo=pytz.UTC)
    cst = timezone('America/Costa_Rica')
    cst_dt = aware_utc_dt.astimezone(cst)
    return cst_dt

def num_examples(dataset, example_type):
    """
    Counts num. positive (1) or negative (0) elements of dataset.
    """
    examples_list = []
    for element in dataset:
        if element[1] == example_type:
            examples_list.append(element[1])
    return len(examples_list)

def clip_long_into_snippets(file_name, duration):
    """
    Clips file into sections of specified length (in seconds).
    Stores clips in custom-labelled folder within 'snippets' directory.
    """
    isolated_name = os.path.basename(os.path.splitext(file_name)[0])
    clip_dir = '/home/dgabutler/Work/CMEEProject/Data/snippets/'+isolated_name+'/'
    if not os.path.exists(clip_dir):
        os.makedirs(clip_dir)
    try:
        wavfile = AudioSegment.from_wav(file_name)
    except OSError:
        print("\nerror: audio file",os.path.basename(file_name),"at path", os.path.dirname(file_name), "cannot be loaded - either does not exist or was improperly recorded")
        return 

    # make clips
    clips = make_chunks(wavfile, duration*1000)
    for idx, clip in enumerate(clips):
        clip_name = isolated_name+'_'+"clip{0:02}.wav".format(idx+1)
        clip.export(clip_dir+clip_name, format="wav")

def whinny_starttimes_from_praatfile(praat_file):
    """
    Extracts whinny start times (in milliseconds) from praat text file
    Output is tuple containing .wav file name and then all times (in ms)
    """
    start_times = []    # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()
    
    # - WAVNAME NOW FOUND USING NAME OF PRAAT FILE, NOT BY READING PRAAT FILE
    # line_with_wavname = praat_contents[10]
    # result = re.search('"(.*)"', line_with_wavname)
    # wav_name = result.group(1)

    wav_name = os.path.basename(os.path.splitext(praat_file)[0])

    for idx, line in enumerate(praat_contents): 
        if "Whinny" in line and "intervals" in praat_contents[idx+1]:
            time_line = praat_contents[idx-2]
            start_times.extend(re.findall("(?<=xmin\s=\s)(\d+\.?\d*)(?=\s)", time_line))

    start_times = map(float, start_times) # converts time to number, not 
                                          # character string
    
    # Comment line below if time in seconds is wanted:
    start_times = [times * 1000 for times in start_times]

    return wav_name, start_times

def whinny_endtimes_from_praatfile(praat_file):
    """
    Extracts whinny end times (in milliseconds) from praat text file
    Output is tuple containing .wav file name and then all times in ms
    """
    end_times = []    # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()
    
    # - WAVNAME NOW FOUND USING NAME OF PRAAT FILE, NOT BY READING PRAAT FILE
    # line_with_wavname = praat_contents[10]
    # result = re.search('"(.*)"', line_with_wavname)
    # wav_name = result.group(1)

    wav_name = os.path.basename(os.path.splitext(praat_file)[0])

    for idx, line in enumerate(praat_contents): 
        if "Whinny" in line and "intervals" in praat_contents[idx+1]:
            time_line = praat_contents[idx-1]
            end_times.extend(re.findall("\d+\.\d+", time_line))
    
    end_times = map(float, end_times) # converts time to number, not 
                                      # character string
    
    # Comment line below if time in seconds is wanted:
    end_times = [times * 1000 for times in end_times]
    
    return wav_name, end_times

def non_starttimes_from_praatfile(praat_file):
    """
    Extracts start time of region known to not contain a spider 
    monkey whinny from praat text file (for generating negatives)
    Output is tuple containing .wav file name and then all times (in ms)
    """
    start_times = []    # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()
    
    line_with_wavname = praat_contents[10]
    result = re.search('"(.*)"', line_with_wavname)
    wav_name = result.group(1)

    for idx, line in enumerate(praat_contents): 
        if "Non Call" in line:
            time_line = praat_contents[idx-2]
            start_times.extend(re.findall("\d+\.\d+|\d", time_line))

    start_times = map(float, start_times) # converts time to number, not 
                                          # character string
    
    # Comment line below if time in seconds is wanted:
    start_times = [times * 1000 for times in start_times]

    return wav_name, start_times

def non_endtimes_from_praatfile(praat_file):
    """
    Extracts end time of region known to not contain a spider 
    monkey whinny from praat text file (for generating negatives)
    Output is tuple containing .wav file name and then all times in ms
    """
    end_times = []    # empty list to store any hits
    with open(praat_file, "rt") as f:
        praat_contents = f.readlines()
    
    line_with_wavname = praat_contents[10]
    result = re.search('"(.*)"', line_with_wavname)
    wav_name = result.group(1)

    for idx, line in enumerate(praat_contents): 
        if "Non Call" in line:
            time_line = praat_contents[idx-1]
            end_times.extend(re.findall("\d+\.\d+|\d{2}", time_line))
    
    end_times = map(float, end_times) # converts time to number, not 
                                      # character string
    
    # Comment line below if time in seconds is wanted:
    end_times = [times * 1000 for times in end_times]
    
    return wav_name, end_times

def clip_whinnies(praat_files, desired_duration, unclipped_folder_location, clipped_folder_location):
    """
    Clip all whinnies from wav files, following labels in praat files, to specified length.

    praat_files = sorted(os.listdir('../Data/praat-files'))

    unclipped_folder_location example: '../Data/unclipped-whinnies'
    clipped_folder_location example: '../Data/clipped-whinnies-osa'

    """

    for file in praat_files:

        start_times = whinny_starttimes_from_praatfile('../Data/praat-files/'+file)
        end_times = whinny_endtimes_from_praatfile('../Data/praat-files/'+file)
        wav_name = end_times[0]

        # following try-except accounts for praat files missing corresponding
        # audio files
        try:
            wavfile = AudioSegment.from_wav(unclipped_folder_location + '/' + wav_name + '.WAV')
        except IOError:
            # print("error: no wav file named " + wav_name + ".WAV at path " + unclipped_folder_location)
            continue

        # desired_duration = 3000 # in milliseconds

        for idx, time in enumerate(end_times[1]): 
            if (len(wavfile) - time >= desired_duration):
                clip_start = start_times[1][idx] # add - 200 to start 0.2 sec in
                clip_end = clip_start + desired_duration
                clip = wavfile[clip_start:clip_end]
            else:          
                # i.e. if start time of a call to end of file doesn't 
                # leave enough for a whole clip: 
                clip = wavfile[-desired_duration:]  # last 'duration' seconds
                                                    # of file 
            
            # Save clipped file to separate folder
            clip.export(clipped_folder_location+'/'+wav_name+'_'+str(idx+1)+'.WAV', format="wav")

def clip_noncall_sections(praat_files):
    """
    Clip out all sections, of variable lengths, labelled 'non call'.
    Input is list of all praat files, generated using:
    praat_files = sorted(os.listdir('/home/dgabutler/Work/CMEEProject/Data/praat-files'))
    """
    for file in praat_files:
    
        start_times = non_starttimes_from_praatfile('/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file)
        end_times = non_endtimes_from_praatfile('/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file)
        wav_name = end_times[0]

        # Following try-except accounts for praat files missing corresponding
        # audio files
        try:
            wavfile = AudioSegment.from_wav('/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/'+wav_name+'.WAV')

        except IOError:
            print("error: no wav file named",wav_name,".WAV at path /home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies")
            continue

        for idx, time in enumerate(end_times[1]): 

            clip_start = start_times[1][idx] 
            clip_end = end_times[1][idx] 

            clip = wavfile[clip_start:clip_end]

            # Save clipped file to separate folder
            clip.export('/home/dgabutler/Work/CMEEProject/Data/sections-without-whinnies/'+wav_name+'_'+str(idx+1)+'.WAV', format="wav")

def generate_negative_examples(noncall_files, desired_length):
    """
    Creates clips of desired length from all files known to not 
    contain monkey whinnies.
    Input is list of all noncall files, generated using:
    noncall_files = sorted(os.listdir('/home/dgabutler/Work/CMEEProject/Data/sections-without-whinnies'))
    """
    for idx, file in enumerate(noncall_files):
        wavfile = AudioSegment.from_wav('/home/dgabutler/Work/CMEEProject/Data/sections-without-whinnies/'+file)
        if wavfile.duration_seconds < desired_length: continue 
        else:            
            clip = wavfile[:desired_length*1000] 
        # Save clipped file to separate folder
        clip.export('/home/dgabutler/Work/CMEEProject/Data/clipped-negatives/'+file, format="wav")

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
    data_folder_path = '/home/dgabutler/Work/CMEEProject/Data/'
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
    data_folder_path = '/home/dgabutler/Work/CMEEProject/Data/'
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

def standardise_inputs(dataset):
    """uses standardisation method from 
    https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    , applies to all spects in dataset
    NB! load_mag_spec also has normalizing element,
    taken from answer to stack overflow Q by Stefan Kahl
    """
    for row in dataset:
        row = list(row)
        # row[0] /= np.max(np.abs(row[0]),axis=0)
        row[0] /= np.max(np.abs(row[0]),axis=None)
    return dataset

# not useful

# clipping whinnies --
praat_files = sorted(os.listdir('../Data/praat-files'))

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


###### AUGMENTATION FUNCTIONS

# useful:

def augment_time_shift(file_name, folder, desired_duration, min_overlap, num_augmentations_per_clip, destination_folder='aug-timeshifted'):
    """
    For a given file name (without extension), creates clips in which
    monkey call is present for at least a minimum of 'min_overlap'
    percentage. I.e. '0.05' would ensure that at least 5% of call
    is present, either at the beginning or the end.

    Example call, try with file name '5A3AE4BC'

    """
    start_times = whinny_starttimes_from_praatfile('/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file_name+'.TextGrid')[1]
    end_times = whinny_endtimes_from_praatfile('/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file_name+'.TextGrid')[1]
    
    # following try-except accounts for praat files missing corresponding
    # audio files
    try:
        wav = AudioSegment.from_wav('/home/dgabutler/Work/CMEEProject/Data/'+folder+'/'+file_name+'.WAV')

    except IOError:
        print 'warning: no wav file named ' +file_name+ '.WAV in folder ' + folder
        return

    call_durations = [a - b for a, b in zip(end_times,start_times)]

    for idx, start_time in enumerate(start_times):

        i = 0
        while i < num_augmentations_per_clip:

            ### randomly define clip start and end times
            start_range_lower_limit = start_time - desired_duration + (min_overlap*call_durations[idx])
            start_range_upper_limit = end_times[idx] - (min_overlap*call_durations[idx])

            clip_start = start_range_lower_limit + random.uniform(0,1)*(start_range_upper_limit - start_range_lower_limit)

            if clip_start < 0:
                clip_start = 0
                
            clip_end = clip_start + desired_duration
            
            if clip_end > 60000:
                clip_end = 60000
                clip_start = clip_end - desired_duration

            ### clip wav file at those random points and save
            clip = wav[clip_start:clip_end] 
            clip.export('/home/dgabutler/Work/CMEEProject/Data/'+destination_folder+'/'+file_name+'_'+str(idx+1)+'_'+str(i+1)+'.WAV', format="wav")
            
            ### iterative counter in while loop
            i+=1    

def augment_folder_time_shift(folder, min_overlap, num_augmentations_per_clip, destination_folder='aug-timeshifted'):
    """
    Applies time shift augment to all clipped positives.
    Clips augmented for average specified number of times e.g. 3 will on average produce 3, but sometimes 2, 4, 0 etc., introducing element of randomness.

    Clipped positives from /Data/praat-files 

    'min_overlap' percentage: '0.05' would ensure that at least 5% of call
    is present, either at the beginning or the end.
    """
    praat_files = sorted(os.listdir('/home/dgabutler/Work/CMEEProject/Data/praat-files'))
    file_names = [os.path.splitext(x)[0] for x in praat_files]
    for file in file_names:
        augment_time_shift(file, folder, 3000, min_overlap, num_augmentations_per_clip, destination_folder)

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

# potentially not useful:

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