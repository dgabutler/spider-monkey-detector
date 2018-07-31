# Exploring functions from Kahl et al. 2017 e.g. making spects
# Machine learning library: Lasagne, with theano
# Date: 10.07.18

# Will be run from code directory in project i.e. 
# /home/dgabutler/CMEECourseWork/Project/Code/
# HOWEVER: at present, no relative paths

# Exploring _spec.py
import os
import traceback
import operator

import numpy as np
import cv2

import scipy.io.wavfile as wave
import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy import interpolate

import python_speech_features as psf
from pydub import AudioSegment

# change sample rate if not 44.1kHz
def changeSampleRate(sig, rate):

    duration = sig.shape[0] / rate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * 44100 / rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    sig = np.round(new_audio).astype(sig.dtype)
    
    return sig, 44100

# get magnitude spec from signal split
def getMagSpec(sig, rate, winlen, winstep, NFFT):

    # get frames
    winfunc = lambda x:np.ones((x,))
    frames = psf.sigproc.framesig(sig, winlen*rate, winstep*rate, winfunc)        
    
    # magnitude spectrogram
    magspec = np.rot90(psf.sigproc.magspec(frames, NFFT))

    return magspec

# split signal into five-second chunks with overlap of 4 and minimum length of 3 seconds
# use these settings for other chunk lengths:
# winlen, winstep, seconds
#0.05, 0.0097, 5s
#0.05, 0.0195, 10s
#0.05, 0.0585, 30s
def get_multi_spec(path, seconds=3, overlap=2, minlen=3, winlen=0.05, winstep=0.00585, NFFT=840):

    # open wav file
    (rate,sig) = wave.read(path)
    print "SampleRate:", rate,

    # adjust to different sample rates
    if rate != 44100:
        sig, rate = changeSampleRate(sig, rate)

    # split signal with ovelap
    sig_splits = []
    for i in xrange(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + seconds * rate]
        if len(split) >= minlen * rate:
            sig_splits.append(split)

    # is signal too short for segmentation?
    # append it anyway
    if len(sig_splits) == 0:
        sig_splits.append(sig)

    # calculate spectrogram for every split
    for sig in sig_splits:

        #preemphasis
        sig = psf.sigproc.preemphasis(sig, coeff=0.95)

        #get spec
        magspec = getMagSpec(sig, rate, winlen, winstep, NFFT)

        #get rid of high frequencies
        h, w = magspec.shape[:2]
        magspec = magspec[h - 256:, :]

        #normalize in [0, 1]
        magspec -= magspec.min(axis=None)
        magspec /= magspec.max(axis=None)        

        #fix shape to 512x256 pixels without distortion
        magspec = magspec[:256, :512]
        temp = np.zeros((256, 512), dtype="float32")
        temp[:magspec.shape[0], :magspec.shape[1]] = magspec
        magspec = temp.copy()
        magspec = cv2.resize(magspec, (512, 256))
        
        #DEBUG: show spec
        #cv2.imshow('SPEC', magspec)
        #cv2.waitKey(-1)

        yield magspec

# remove single spots from an image
def filter_isolated_cells(array, struct):

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    
    return filtered_array

# decide if given spectrum shows noise of interest or background sound only
def has_monkey(spec, threshold=16):

    # working copy
    img = spec.copy()

    # STEP 1: Median blur
    img = cv2.medianBlur(img,5)

    # STEP 2: Median threshold
    col_median = np.median(img, axis=0, keepdims=True)
    row_median = np.median(img, axis=1, keepdims=True)

    img[img < row_median * 3] = 0
    img[img < col_median * 4] = 0
    img[img > 0] = 1

    # STEP 3: Remove singles
    img = filter_isolated_cells(img, struct=np.ones((3,3)))

    # STEP 4: Morph Closing
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5), np.float32))

    # STEP 5: Frequency crop
    img = img[128:-16, :]

    # STEP 6: Count columns and rows with signal
    # (Note: We only use rows with signal as threshold, but columns might come in handy in other scenarios)

    # column has signal?
    col_max = np.max(img, axis=0)
    col_max = ndimage.morphology.binary_dilation(col_max, iterations=2).astype(col_max.dtype)
    cthresh = col_max.sum()

    # row has signal?
    row_max = np.max(img, axis=1)
    row_max = ndimage.morphology.binary_dilation(row_max, iterations=2).astype(row_max.dtype)
    rthresh = row_max.sum()

    # final threshold
    thresh = rthresh

    #DBUGB: show?
    #print thresh
    #cv2.imshow('BIRD?', img)
    #cv2.waitKey(-1)

    # STEP 7: Apply threshold (Default = 16)
    noise_of_interest = True
    if thresh < threshold:
        noise_of_interest = False
    
    return noise_of_interest, thresh

######################################################

IM_AUGMENTATION = {#'type':[probability, value]
                   'roll':[0.5, (0.0, 0.05)], 
                   'noise':[0.1, 0.01],
                   'noise_samples':[0.1, 1.0],
                   #'brightness':[0.5, (0.25, 1.25)],
                   #'crop':[0.5, 0.07],
                   #'flip': [0.25, 1]
                   }

#gaussian noise
if 'noise' in AUG and RANDOM.choice([True, False], p=[AUG['noise'][0], 1 - AUG['noise'][0]]):
    img += RANDOM.normal(0.0, RANDOM.uniform(0, AUG['noise'][1]**0.5), img.shape)
    img = np.clip(img, 0.0, 1.0)

#add noise samples
if 'noise_samples' in AUG and RANDOM.choice([True, False], p=[AUG['noise_samples'][0], 1 - AUG['noise_samples'][0]]):
    img += openImage(NOISE[RANDOM.choice(range(0, len(NOISE)))], True) * AUG['noise_samples'][1]
    img -= img.min(axis=None)
    img /= img.max(axis=None)


#elist all bird species
birds = [src_dir + bird + '/' for bird in sorted(os.listdir(src_dir))][:MAX_SPECIES]
print "BIRDS:", len(birds)

#parse bird species
for bird in birds:
    total_specs = 0
    
    #get all wav files
    wav_files = [bird + wav for wav in sorted(os.listdir(bird))]

    #parse wav files
    for wav in wav_files:
        spec_cnt = 0
        print wav,
        
        try:
            #get every spec from each wav file
            for spec in getMultiSpec(wav):

                #does spec contain bird sounds?
                isbird, thresh = hasBird(spec)

                #new target path -> rejected specs will be copied to "noise" folder
                if isbird:
                    dst_dir = spec_dir + bird.split("/")[-2] + "/"
                else:
                    dst_dir = spec_dir + "noise/" + bird.split("/")[-2] + "/"

                #make target dir
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                    
                #write spec to target dir
                cv2.imwrite(dst_dir +  str(thresh) + "_" + wav.split("/")[-1].rsplit(".")[0] + "_" + str(spec_cnt) + ".png", spec * 255.0)
                
                spec_cnt += 1
                total_specs += 1

            #exceeded spec limit?
            if total_specs >= MAX_SPECS and MAX_SPECS > -1:
                print " "
                break
            
            print "SPECS:", spec_cnt

        except:
            print spec_cnt, "ERROR"
            traceback.print_exc()
            pass            


