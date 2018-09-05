# ALL RUNNING CODE FOR CMEE MSC PROJECT
# - includes misc. data wrangling (e.g. clipping of 
# specific non-monkey noises from files) 
# Date: 21.07.18

import os 
import sys
import csv 
import glob
import random
import pickle
import time
import glob
import os

sys.path.insert(0, '/home/dgabutler/Work/CMEEProject/Code')
import wavtools   # contains custom functions e.g. denoising
import keras_classifier

###### DATA PROCESSING
# check target folders are empty, if script is re-run:
# clipped folders:
folders = glob.glob('../Data/clipped-whinnies-*/')
for folder in folders:
    files = glob.glob(folder+'*.WAV')
    for f in files:
        os.remove(f)
# augmented folder:
files = glob.glob('../Data/aug-timeshifted/*.WAV')
for f in files:
    os.remove(f)
#
# clipping positives:
praat_files = sorted(os.listdir('../Data/praat-files'))
# LOCATION 1: OSA
# positives:
wavtools.clip_whinnies(praat_files, 3000, '../Data/unclipped-whinnies-osa','../Data/clipped-whinnies-osa')
# LOCATION 2: SHADY
# positives:
wavtools.clip_whinnies(praat_files, 3000, '../Data/unclipped-whinnies-shady','../Data/clipped-whinnies-shady')
# LOCATION 3: CORCOVADO
# positives:
wavtools.clip_whinnies(praat_files, 3000, '../Data/unclipped-whinnies-corcovado','../Data/clipped-whinnies-corcovado')
# LOCATION 4: CATAPPA
# positives:
wavtools.clip_whinnies(praat_files, 3000, '../Data/unclipped-whinnies-catappa','../Data/clipped-whinnies-catappa')

# TIME-SHIFT AUGMENT:
#
wavtools.augment_folder_time_shift('unclipped-whinnies-osa', 0.2, 2)
wavtools.augment_folder_time_shift('unclipped-whinnies-catappa', 0.2, 2)
wavtools.augment_folder_time_shift('unclipped-whinnies-shady', 0.2, 2)


###### LOADING TRAINED MODEL
loaded_model = keras_classifier.load_keras_model('D_mined_aug_tb_denoised', 'e50_b16')

#################################################################
###################### -- WRANGLING -- ##########################

praat_files = sorted(os.listdir('/home/dgabutler/Work/CMEEProject/Data/praat-files'))

# ########## HARD NEGATIVE MINING

# # keras_classifier.hard_negative_miner(wav_folder='/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/', threshold_confidence=70)

# #################################################################
# ##################### -- AUGMENTING -- ##########################

# ################### RANDOM TIME SHIFT ###########

# file_names = [os.path.splitext(x)[0] for x in praat_files]
# for file in file_names:
#     wavtools.augment_time_shift(file, 3000, 0.3, 1)

# ################### FILE BLENDING ################

# # - wrangling non-monkey noises 
# # - file names from excel sheet
# # file_list = []
# # with open('/home/dgabutler/Work/CMEEProject/Data/misc/file-names-to-mine-for-non-monkey-noises.csv', 'rb') as f:
# #     reader = csv.reader(f)
# #     file_list = list(reader)
# # files_to_mine = [item for sublist in file_list for item in sublist]

# # wav_folder = '/media/dgabutler/My Passport/Audio/Catappa/'
# # wavtools.search_file_list_for_monkeys(files_to_mine, wav_folder=wav_folder, threshold_confidence=60)

# # - deliberate clipping
# # - howler monkey:
# # wavtools.clip_long_into_snippets('/media/dgabutler/My Passport/Audio/Catappa/5A3CE5B0.WAV', 3)
# # # move four good clips to folder:
# # from shutil import copyfile
# # data_path = '/home/dgabutler/Work/CMEEProject/Data'
# # for i in xrange(4):
# #     copyfile(data_path+'/snippets/5A3CE5B0/5A3CE5B0_clip0'+str(i+6)+'.wav', data_path+'/non-monkey-noises/5A3CE5B0_howler'+str(i+1)+'.WAV')

# whinny_files = glob.glob('/home/dgabutler/Work/CMEEProject/Data/clipped-whinnies/*.WAV')
# # for whinny in whinny_files:
# #     wavtools.augment_file_blend(whinny)

# #################################################################
# ################### -- DATABASE ASSEMBLY -- #####################

# ## making databases

# ### method 1: with no alterations to the data

# dataset 
D_original = [] 

# - ADD POSITIVES - 
# 1) generate positive clips
wavtools.clip_whinnies(praat_files, 3)
# 2) add clips to dataset
wavtools.add_files_to_dataset(folder='clipped-whinnies', dataset=D_original, example_type=1)
# - ADD NEGATIVES - 
# 1) generate negative clips
# a) populate folder with sections of various lengths known to not contain calls
# wavtools.clip_noncall_sections(praat_files)
# b) clip the beginning of each of these into 3 second clips
# noncall_files = sorted(os.listdir('/home/dgabutler/Work/CMEEProject/Data/sections-without-whinnies'))
# wavtools.generate_negative_examples(noncall_files, 3.00)
# 2) add negative clips to dataset
wavtools.add_files_to_dataset(folder='clipped-negatives', dataset=D_original, example_type=0)

print("\nNumber of samples currently in original dataset: " + str(wavtools.num_examples(D_original,0)) + \
" negative, " + str(wavtools.num_examples(D_original,1)) + " positive")

# method 2: applying denoising to the spectrograms

D_denoised = wavtools.denoise_dataset(D_original)

# method 3: adding augmented (time-shifted) data

D_aug_t = D_original 

wavtools.add_files_to_dataset(folder='aug-timeshifted', dataset=D_aug_t, example_type=1)

print("\nNumber of samples when positives augmented (time): " + str(wavtools.num_examples(D_aug_t,0)) + \
" negative, " + str(wavtools.num_examples(D_aug_t,1)) + " positive")

# method 3.5: adding augmented (blended) data

D_aug_tb = D_aug_t

wavtools.add_files_to_dataset(folder='aug-blended', dataset=D_aug_tb, example_type=1)

print("\nNumber of samples when positives augmented (time shift and blended): " + str(wavtools.num_examples(D_aug_tb,0)) + \
" negative, " + str(wavtools.num_examples(D_aug_tb,1)) + " positive")

# method 4: augmented (both) and denoised

D_aug_t_denoised = wavtools.denoise_dataset(D_aug_t)
D_aug_tb_denoised = wavtools.denoise_dataset(D_aug_tb)

# method 5: adding hard-negative mined training examples 

# keras_classifier.hard_negative_miner('/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/', 62, model=loaded_model)
D_mined_aug_tb = D_aug_tb 
wavtools.add_files_to_dataset(folder='mined-false-positives', dataset=D_mined_aug_tb, example_type=0)

print("\nNumber of samples when hard negatives added: " + str(wavtools.num_examples(D_mined_aug_tb,0)) + \
" negative, " + str(wavtools.num_examples(D_mined_aug_tb,1)) + " positive")

D_mined_aug_tb_denoised = wavtools.denoise_dataset(D_mined_aug_tb)

# method 6: adding selected obvious false positives as training examples

D_S_mined_aug_t_denoised = D_aug_t

wavtools.add_files_to_dataset(folder='selected-false-positives', dataset=D_S_mined_aug_t_denoised, example_type=0)

print("\nNumber of samples when select negatives added: " + str(wavtools.num_examples(D_S_mined_aug_t_denoised,0)) + \
" negative, " + str(wavtools.num_examples(D_S_mined_aug_t_denoised,1)) + " positive")


# #################################################################
# ####################### -- TRAINING -- ##########################

# # (NB. already have model saved, running below will overwrite)
keras_classifier.train_simple_keras(D_S_mined_aug_t_denoised,'D_S_mined_aug_t_denoised',0.85, num_epochs=20, batch_size=64)

# # pulling to one side 1000 files from each of 3 folders, 
# # running search_file_list_for_monkeys on them:

# def filenames_only_from_whole_paths(whole_paths_list):
#     file_names = [os.path.splitext(x)[0] for x in whole_paths_list]
#     file_names = [os.path.basename(x) for x in file_names]
#     return file_names 

# hard_drive = '/media/dgabutler/My Passport/Audio/'
# # rand_1000_catappa2 = filenames_only_from_whole_paths(random.sample(glob.glob(hard_drive+'Catappa May 18/Catappa 2/*WAV'), 1000))
# # rand_1000_osa1 = filenames_only_from_whole_paths(random.sample(glob.glob(hard_drive+'Osa/Osa 1/*WAV'), 1000))
# # rand_1000_teak1 = filenames_only_from_whole_paths(random.sample(glob.glob(hard_drive+'Teak 1/*WAV'), 1000))
# # rand_1000_teak2 = filenames_only_from_whole_paths(random.sample(glob.glob(hard_drive+'Teak 2/*WAV'), 1000))

# # random_files_scanned = []
# # random_files_scanned.append(rand_1000_catappa2)
# # random_files_scanned.append(rand_1000_osa1)
# # random_files_scanned.append(rand_1000_teak1)
# # random_files_scanned.append(rand_1000_teak2)

# # with open('/home/dgabutler/Work/CMEEProject/Data/misc/list_of_random_files_scanned.pkl', 'wb') as fp:
# #     pickle.dump(random_files_scanned, fp)

# # with open('/home/dgabutler/Work/CMEEProject/Data/misc/list_of_random_files_scanned.pkl', 'rb') as fp:
# #     random_files = pickle.load(fp)

# ## testing:
# # dummy_listoflists = []
# # dummy1 = filenames_only_from_whole_paths(glob.glob('/media/dgabutler/My Passport/Audio/dummy1/*WAV'))
# # dummy2 = filenames_only_from_whole_paths(glob.glob('/media/dgabutler/My Passport/Audio/dummy2/*WAV'))
# # dummy3 = filenames_only_from_whole_paths(glob.glob('/media/dgabutler/My Passport/Audio/dummy3/*WAV'))
# # dummy_listoflists.append(dummy1)
# # dummy_listoflists.append(dummy2)
# # dummy_listoflists.append(dummy3)

# # tic = time.time()
# # i = 0
# # # for sublist in dummy_listoflists:
# # for sublist in random_files_scanned:
# #     if i == 0:
# #         # wav_folder = '/media/dgabutler/My Passport/Audio/dummy1/'
# #         wav_folder = '/media/dgabutler/My Passport/Audio/Catappa May 18/Catappa 2/'
# #     elif i == 1:
# #         # wav_folder = '/media/dgabutler/My Passport/Audio/dummy2/'
# #         wav_folder = '/media/dgabutler/My Passport/Audio/Osa/Osa 1/'
# #     elif i == 2:
# #         # wav_folder = '/media/dgabutler/My Passport/Audio/dummy3/'   
# #         wav_folder = '/media/dgabutler/My Passport/Audio/Teak 1/'

# #     keras_classifier.search_file_list_for_monkeys(sublist, wav_folder, threshold_confidence=60, model=loaded_model)
# #     i+=1

# # toc = time.time()
# # total_time = toc-tic # = 3189.8217131628265

# # # run on teak 2, as teak 1 was too fuzzy
# # tic = time.time()
# # keras_classifier.search_file_list_for_monkeys(rand_1000_teak2, '/media/dgabutler/My Passport/Audio/Teak 2/', threshold_confidence=60, model=loaded_model)
# # toc = time.time()
# # total_time = toc-tic 

#################################################################
###################### -- PREDICTING -- #########################

wav_folder = '/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/shady-lane/'
keras_classifier.search_file_for_monkeys(file_name, wav_folder, threshold_confidence)

keras_classifier.search_folder_for_monkeys('/home/dgabutler/Work/CMEEProject/Data/unclipped-whinnies/', 95, model=loaded_model)
# keras_classifier.search_folder_for_monkeys(wav_folder=wav_folder,threshold_confidence=80)

#################################################################
######################## -- MISC. -- ############################

############# PROFILING TO SEE EFFICIENCY
# ran 'python -m cProfile -o profiling_results basic_CNN_\(urbansound\).py' in bash terminal

# import pstats
# stats = pstats.Stats("profiling_results")
# stats.sort_stats("tottime")

# stats.print_stats(30)

#################################################

# reading back in pickle file:
# see https://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python

