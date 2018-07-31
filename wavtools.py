#!/usr/bin/env python

# Functions and scripts for processing .wav files and praat files
# includes: - clipping 60 second files into short sections containing calls
#           - obtaining time of recording from file name 

# date: 21.07.18

# NB. no relative paths in script

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
    
    line_with_wavname = praat_contents[10]
    result = re.search('"(.*)"', line_with_wavname)
    wav_name = result.group(1)

    for idx, line in enumerate(praat_contents): 
        if "_Whinny" in line and "intervals" in praat_contents[idx+1]:
            time_line = praat_contents[idx-2]
            start_times.extend(re.findall("\d+\.\d+", time_line))

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
    
    line_with_wavname = praat_contents[10]
    result = re.search('"(.*)"', line_with_wavname)
    wav_name = result.group(1)

    for idx, line in enumerate(praat_contents): 
        if "_Whinny" in line and "intervals" in praat_contents[idx+1]:
            time_line = praat_contents[idx-1]
            end_times.extend(re.findall("\d+\.\d+", time_line))
    
    end_times = map(float, end_times) # converts time to number, not 
                                      # character string
    
    # Comment line below if time in seconds is wanted:
    end_times = [times * 1000 for times in end_times]
    
    return wav_name, end_times

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

def clip_whinnies(praat_files, desired_duration):
    """
    Clip all whinnies from wav files, following labels in praat files, to specified length.

    praat_files = sorted(os.listdir('/home/dgabutler/Work/CMEEProject/Data/praat-files'))

    """
    unclipped_folder = '/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies'
    clipped_folder_whinnies = '/home/dgabutler/Work/CMEEProject/Data/clipped-whinnies'

    for file in praat_files:

        start_times = whinny_starttimes_from_praatfile('/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file)
        end_times = whinny_endtimes_from_praatfile('/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file)
        wav_name = end_times[0]

        # Following try-except accounts for praat files missing corresponding
        # audio files
        try:
            wavfile = AudioSegment.from_wav(unclipped_folder + '/' + wav_name + '.WAV')
        except IOError:
            print "error: no wav file named " + wav_name + ".WAV at path " + unclipped_folder
            continue

        # desired_duration = 3000 # in milliseconds

        for idx, time in enumerate(end_times[1]): 
            if (60000 - time >= desired_duration):
                clip_start = start_times[1][idx] - 200 # start 0.3 sec in
                clip_end = clip_start + desired_duration
                clip = wavfile[clip_start:clip_end]
            else:          
                # i.e. if start time of a call to end of file doesn't 
                # leave enough for a whole clip: 
                clip = wavfile[-desired_duration:]  # last 'duration' seconds
                                                    # of file 
            
            # Save clipped file to separate folder
            clip.export(clipped_folder_whinnies+'/'+wav_name+'_'+str(idx+1)+'.WAV', format="wav")

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
            print "error: no wav file named",wav_name,".WAV at path /home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies"
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

def add_files_to_dataset(folder, dataset, example_type):
    """
    Takes all wav files from given folder name (minus slashes on 
    either side) and adds to the dataset name provided.
    Example type = 0 if negative, 1 if positive.
    """
    data_folder_path = '/home/dgabutler/Work/CMEEProject/Data/'
    files = glob.glob(data_folder_path+folder+'/*.WAV')
    for wav in files:
        y, sr = librosa.load(wav, sr=None, duration=3.00)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 282): continue
        dataset.append( (ps, example_type) )

def denoise_dataset(dataset):
    """
    Applies denoise function over all spectrograms in dataset
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

###### AUGMENTATION FUNCTIONS

def augment_time_shift(file_name, desired_duration, min_overlap, approx_num_augmentations):
    """
    For a given file name (without extension), creates clips in which
    monkey call is present for at least a minimum of 'min_overlap'
    percentage. I.e. '0.05' would ensure that at least 5% of call
    is present, either at the beginning or the end.
    Number of augmented clips created also has randomness, being 
    approximately the number provided but not guaranteed.

    Example call, try with file name '5A3AE4BC'

    """
    start_times = whinny_starttimes_from_praatfile('/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file_name+'.TextGrid')[1]
    end_times = whinny_endtimes_from_praatfile('/home/dgabutler/Work/CMEEProject/Data/praat-files/'+file_name+'.TextGrid')[1]
    
    # following try-except accounts for praat files missing corresponding
    # audio files
    try:
        wav = AudioSegment.from_wav('/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/'+file_name+'.WAV')

    except IOError:
        print 'error: no wav file named',file_name,'.WAV at path /home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/'
        return

    call_durations = [a - b for a, b in zip(end_times,start_times)]

    while_threshold = 1-1.0/(approx_num_augmentations)

    for idx, start_time in enumerate(start_times):

        RANDOM = -1
        i = 1

        while RANDOM < while_threshold:
        # i.e. should get around specified number of augs, but it varies

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
            clip.export('/home/dgabutler/Work/CMEEProject/Data/aug-timeshifted/'+file_name+'_'+str(idx+1)+'_'+str(i)+'.WAV', format="wav")
            
            ### iterative counters in while loop
            i+=1    # tracks num. augmentations per file
            RANDOM = random.uniform(0,1)

def augment_folder_time_shift(min_perc_overlap, avg_augs_per_clip):
    """
    Applies time shift augment to all clipped positives.
    Clips augmented for average specified number of times e.g. 3 will on average produce 3, but sometimes 2, 4, 0 etc., introducing element of randomness.

    Clipped positives from /Data/praat-files 
    """
    praat_files = sorted(os.listdir('/home/dgabutler/Work/CMEEProject/Data/praat-files'))
    file_names = [os.path.splitext(x)[0] for x in praat_files]
    for file in file_names:
        augment_time_shift(file, 3000, min_perc_overlap, avg_augs_per_clip)

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

############ FOR KERAS CONVNET # this will be needed if I used files sampled at different rates, currently not needed
def calc_time_steps(file, duration=None, sr=None):
    """
    For a given duration and sampling rate, gives number of time steps a file will be broken into
    Solves problem of varying ConvNet input dimension dependent on sampling rate/duration
    """
    y, sr = librosa.load(file, sr=sr, duration=duration)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)

    return ps.shape[1]

#######################################################################################
################ MORE ADVANCED FUNCTIONS FROM MAC AODHA ET AL. 2017
################ - denoises spectrograms, generates features in sliding window over .wav file
################ - NB. ONLY DENOISE IS USED BY CODE SO FAR. THE REST IS YET TO BE IMPLIMENTED

def denoise(spec_noisy):
    """
    Subtract mean from each frequency band
    Modified from Mac Aodha et al. 2017
    """
    me = np.mean(spec_noisy, 1)
    spec_denoise = spec_noisy - me[:, np.newaxis]

    # remove anything below 0
    spec_denoise.clip(min=0, out=spec_denoise)

    return spec_denoise

def process_spectrogram(spec, denoise_spec=True, smooth_spec=True):
    """
    Denoises, and smooths spectrogram.
    NB removed mean-log magnitude section, don't think I need it
    as I don't have silence on either side of my recordings
    """

    # denoise
    if denoise_spec:
        spec = denoise(spec)

    # smooth the spectrogram
    if smooth_spec:
        spec = filters.gaussian(spec, 1.0)

    return spec

def gen_mag_spectrogram(x, fs, ms, overlap_perc):
    """
    Computes magnitude spectrogram by specifying time.
    """

    nfft = int(ms*fs)
    noverlap = int(overlap_perc*nfft)

    # window data
    step = nfft - noverlap
    shape = (nfft, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # apply window
    x_wins_han = np.hanning(x_wins.shape[0])[..., np.newaxis] * x_wins

    # do fft
    # note this will be much slower if x_wins_han.shape[0] is not a power of 2
    complex_spec = np.fft.rfft(x_wins_han, axis=0)

    # calculate magnitude
    mag_spec = (np.conjugate(complex_spec) * complex_spec).real
    # same as:
    #mag_spec = np.square(np.absolute(complex_spec))

    # orient correctly and remove dc component
    spec = mag_spec[1:, :]
    spec = np.flipud(spec)

    return spec

def gen_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap, crop_spec=True, max_freq=256, min_freq=0):
    """
    Compute spectrogram, crop and compute log.
    [might need to edit section about max and min frequency???]
    """

    # compute spectrogram
    spec = gen_mag_spectrogram(audio_samples, sampling_rate, fft_win_length, fft_overlap)

    # only keep the relevant bands - could do this outside
    if crop_spec:
        spec = spec[-max_freq:-min_freq, :]

        # add some zeros if too small
        req_height = max_freq-min_freq
        if spec.shape[0] < req_height:
            zero_pad = np.zeros((req_height-spec.shape[0], spec.shape[1]))
            spec = np.vstack((zero_pad, spec))

    # perform log scaling - here the same as matplotlib
    log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(fft_win_length*sampling_rate)))**2).sum())
    spec = np.log(1.0 + log_scaling*spec)

    return spec

def compute_features(audio_samples, sampling_rate, params):
    """
    Computes overlapping windows of spectrogram as input for CNN.
    """

    # load audio and create spectrogram
    spectrogram = gen_spectrogram(audio_samples, sampling_rate, params.fft_win_length, params.fft_overlap,
                                     crop_spec=params.crop_spec, max_freq=params.max_freq, min_freq=params.min_freq)
    spectrogram = process_spectrogram(spectrogram, denoise_spec=params.denoise, smooth_spec=params.smooth_spec)

    # extract windows
    spec_win = view_as_windows(spectrogram, (spectrogram.shape[0], params.window_width))[0]
    spec_win = zoom(spec_win, (1, 0.5, 0.5), order=1)
    spec_width = spectrogram.shape[1]

    # make the correct size for CNN
    features = np.zeros((spec_width, 1, spec_win.shape[1], spec_win.shape[2]), dtype=np.float32)
    features[:spec_win.shape[0], 0, :, :] = spec_win

    return features

### TESTING MORE ADVANCED FUNCTIONS
# import sys 
# sys.path.insert(0, '/home/dgabutler/Work/CMEEProject/Sandbox/batdetect/bat_train')
# from data_set_params import DataSetParams
# from skimage import filters
# from skimage.util.shape import view_as_windows
# from scipy.ndimage import zoom
# from scipy.io import wavfile

# params = DataSetParams()

# sampling_rate, audio_samples = wavfile.read('/home/dgabutler/Work/CMEEProject/Data/clipped-whinnies/5A383EF0_1.WAV')
# features = compute_features(audio_samples, sampling_rate, params=params)