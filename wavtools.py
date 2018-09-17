# Functions for general processing of .wav files 
# e.g. clipping using label files, preprocessing (denoising/normalizing)
# augmenting, and adding to datasets for training of CNNs 

from    datetime import datetime # used by filename_to_localdatetime
from    pytz import timezone # used by filename_to_localdatetime
import  pytz
import  re
import  os
from    pydub import AudioSegment  # used by: aug_time_shift
from    pydub.utils import make_chunks
import  os
import  numpy as np 
import  random # used by: aug_time_shift            
import  glob # used by: aug_file_blend       
import  librosa
import  librosa.display   
from    sklearn.utils import shuffle # used by do_augmentation
import  python_speech_features as psf # used by load_mag_spec

praat_files = sorted(os.listdir('../Data/praat-files'))

#######################
### WRANGLING FUNCTIONS

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
    clip_dir = '../Data/snippets/'+isolated_name+'/'
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

def clip_noncall_sections(praat_files):
    """
    Clip out all sections, of variable lengths, labelled 'non call'.
    Input is list of all praat files, generated using:
    praat_files = sorted(os.listdir('../Data/praat-files'))
    """
    for file in praat_files:
    
        start_times = non_starttimes_from_praatfile('../Data/praat-files/'+file)
        end_times = non_endtimes_from_praatfile('../Data/praat-files/'+file)
        wav_name = end_times[0]

        # Following try-except accounts for praat files missing corresponding
        # audio files
        try:
            wavfile = AudioSegment.from_wav('../Data/unclipped-whinnies/'+wav_name+'.WAV')

        except IOError:
            print("error: no wav file named",wav_name,".WAV at path /home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies")
            continue

        for idx, time in enumerate(end_times[1]): 

            clip_start = start_times[1][idx] 
            clip_end = end_times[1][idx] 

            clip = wavfile[clip_start:clip_end]

            # Save clipped file to separate folder
            clip.export('../Data/sections-without-whinnies/'+wav_name+'_'+str(idx+1)+'.WAV', format="wav")

def generate_negative_examples(noncall_files, desired_length):
    """
    Creates clips of desired length from all files known to not 
    contain monkey whinnies.
    Input is list of all noncall files, generated using:
    noncall_files = sorted(os.listdir('../Data/sections-without-whinnies'))
    """
    for idx, file in enumerate(noncall_files):
        wavfile = AudioSegment.from_wav('../Data/sections-without-whinnies/'+file)
        if wavfile.duration_seconds < desired_length: continue 
        else:            
            clip = wavfile[:desired_length*1000] 
        # Save clipped file to separate folder
        clip.export('../Data/clipped-negatives/'+file, format="wav")

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

###### AUGMENTATION FUNCTIONS
##### NB. only has time-shift, as that is the only augmentation that physically creates new .wav files
##### other two augmentations used (Gaussian noise and background noise blending) applied using 'add_n_files...' function
##### within keras_classifier module

def augment_time_shift(file_name, folder, desired_duration, min_overlap, num_augmentations_per_clip, destination_folder='aug-timeshifted'):
    """
    For a given file name (without extension), creates clips in which
    monkey call is present for at least a minimum of 'min_overlap'
    percentage. I.e. '0.05' would ensure that at least 5% of call
    is present, either at the beginning or the end.

    Example call, try with file name '5A3AE4BC'

    """
    start_times = whinny_starttimes_from_praatfile('../Data/praat-files/'+file_name+'.TextGrid')[1]
    end_times = whinny_endtimes_from_praatfile('../Data/praat-files/'+file_name+'.TextGrid')[1]
    
    # following try-except accounts for praat files missing corresponding
    # audio files
    try:
        wav = AudioSegment.from_wav('../Data/'+folder+'/'+file_name+'.WAV')

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
            clip.export('../Data/'+destination_folder+'/'+file_name+'_'+str(idx+1)+'_'+str(i+1)+'.WAV', format="wav")
            
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
    praat_files = sorted(os.listdir('../Data/praat-files'))
    file_names = [os.path.splitext(x)[0] for x in praat_files]
    for file in file_names:
        augment_time_shift(file, folder, 3000, min_overlap, num_augmentations_per_clip, destination_folder)
