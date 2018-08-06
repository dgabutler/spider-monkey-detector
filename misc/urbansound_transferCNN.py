# Data processing script for Salamon & Bello 2017 CNN,
# trained on massively augmented urbansound8k dataset

import pandas as pd 
import librosa
from scipy import interpolate # used in changeSampleRate
import numpy as np # used in changeSampleRate
import os 
import soundfile as sf 
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

def trim_empty_rows_from_ndarray(ndarray):
    """calcs idx of first row in ndarray that didn't recieve data,
       trims ndarray to contain only all rows up to first empty row.
    """
    end_of_vals = []
    for idx, row in enumerate(ndarray):
        if len(end_of_vals) == 1: break 
        all_zeros = not np.any(row[0])
        if all_zeros:
            end_of_vals.append(idx)
    return ndarray[:end_of_vals[0],:]

# Read Data
data = pd.read_csv('../Data/UrbanSound8K/metadata/UrbanSound8K.csv', delimiter=',', encoding="utf-8-sig")

# Get data over 3 seconds long
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] >= 4 ]

valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
valid_data['augpathspeed1'] = 'fold' + valid_data['fold'].astype('str') + '/' + 'speed_81/' + valid_data['slice_file_name'].astype('str')
valid_data['augpathspeed2'] = 'fold' + valid_data['fold'].astype('str') + '/' + 'speed_107/' + valid_data['slice_file_name'].astype('str')
valid_data['augpathps1'] = 'fold' + valid_data['fold'].astype('str') + '/' + 'ps_200/' + valid_data['slice_file_name'].astype('str')
valid_data['augpathps2'] = 'fold' + valid_data['fold'].astype('str') + '/' + 'ps_250/' + valid_data['slice_file_name'].astype('str')

# Building AUG. directory structure
augs = [1.07, 0.81, 2, 2.5]
for idx, aug in enumerate(augs):
    for i in range(10):
        if idx <= 1:
            aug_type = '/speed_'
        else:
            aug_type = '/ps_'
        dir = '../Data/UrbanSound8K/augmented/fold' + str(i+1) + aug_type + str(int(aug*100))
        if not os.path.exists(dir):
            os.makedirs(dir)

## ADD DATA TO DATASET 
# iterate over all samples in valid. for every sample, 
# construct the (128,259) spectrogram

df = np.zeros(shape=(13000,2), dtype=object) # dataset

# add normal data
df_idx = 0
job_idx = 0
# files = np.zeros(5000)
print('\nnumber of files added (approx. percent of total): ')
for row in valid_data.itertuples():
    job_idx += 1
    try:
        y, sr = sf.read('../Data/UrbanSound8K/audio/' + row.path, stop=132300)
    except IOError: continue
    # files[job_idx] = '../Data/UrbanSound8K/audio/' + row.path
    if sr != 44100: continue
    y = librosa.to_mono(np.ndarray.transpose(y))
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    if ps.shape != (128, 259): continue  
    # add values to df
    df[df_idx] = [ps, row.classID]
    # print statement shows progress
    if df_idx % 200 == 0:
        print(str(df_idx)+' ('+str(round((df_idx/24500*100), 3))+'%)', end='\r')
    # increment dataframe index if value added
    df_idx += 1  
trimmed_df = trim_empty_rows_from_ndarray(df)
joblib.dump(trimmed_df, '../Data/UrbanSound8K/unaugmented-df') 

## VARY TIME

rate = 1.07 # replace with 0.81 and execute again
print('\nnumber of files added (approx. percent of total): ')
for row in valid_data.itertuples():
    y, sr = sf.read('../Data/UrbanSound8K/audio/' + row.path, stop=176400)
    # y, sr = sf.read('../Data/UrbanSound8K/audio/fold5/100263-2-0-117.wav', stop=176400) 
    if sr != 44100: continue
    y = librosa.to_mono(np.ndarray.transpose(y))
    y_changed = librosa.effects.time_stretch(y, rate=rate)
    y_changed = y_changed[:132300]
    ps = librosa.feature.melspectrogram(y=y_changed, sr=sr)
    if ps.shape != (128, 259): continue
    # add values to df
    df[df_idx] = [ps, row.classID]
    # print statement shows progress
    if df_idx % 50 == 0:
        print(str(df_idx)+' ('+str(round((df_idx/24500*100), 3))+'%)', end='\r')    # increment dataframe index if value added
    df_idx += 1  
joblib.dump('../Data/UrbanSound8K/aug1-df.npy', df) 

rate = 0.81 
print('\nnumber of files added (approx. percent of total): ')
for row in valid_data.itertuples():
    y, sr = sf.read('../Data/UrbanSound8K/audio/' + row.path, stop=132300)
    # y, sr = sf.read('../Data/UrbanSound8K/audio/fold5/100263-2-0-117.wav', stop=176400) 
    if sr != 44100: continue
    y = librosa.to_mono(np.ndarray.transpose(y))
    y_changed = librosa.effects.time_stretch(y, rate=rate)
    y_changed = y_changed[:132300]
    ps = librosa.feature.melspectrogram(y=y_changed, sr=sr)
    if ps.shape != (128, 259): continue
    # add values to df
    df[df_idx] = [ps, row.classID]
    # print statement shows progress
    if df_idx % 50 == 0:
        print(str(df_idx)+' ('+str(round((df_idx/24500*100), 3))+'%)', end='\r')    # increment dataframe index if value added
    df_idx += 1  
joblib.dump('../Data/UrbanSound8K/aug2-df.npy', df) 

## VARY PITCH

n_steps = 2 #-1, -2, 2, 1
print('\nnumber of files added (approx. percent of total): ')
for row in valid_data.itertuples():
    y, sr = sf.read('../Data/UrbanSound8K/audio/' + row.path, stop=132300)
    # y, sr = sf.read('../Data/UrbanSound8K/audio/fold5/100263-2-0-117.wav', stop=176400) 
    if sr != 44100: continue
    y = librosa.to_mono(np.ndarray.transpose(y))
    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    y_changed = y_changed[:132300]
    ps = librosa.feature.melspectrogram(y=y_changed, sr=sr)
    if ps.shape != (128, 259): continue
    # add values to df
    df[df_idx] = [ps, row.classID]
    # print statement shows progress
    if df_idx % 50 == 0:
        print(str(df_idx)+' ('+str(round((df_idx/24500*100), 3))+'%)', end='\r')    # increment dataframe index if value added
    df_idx += 1  
joblib.dump(df, '../Data/UrbanSound8K/aug3-df.npy') 

n_steps = 2.5 #-2.5, -3.5, 2.5, 3.5
print('\nnumber of files added (approx. percent of total): ')
for row in valid_data.itertuples():
    y, sr = sf.read('../Data/UrbanSound8K/audio/' + row.path, stop=132300)
    # y, sr = sf.read('../Data/UrbanSound8K/audio/fold5/100263-2-0-117.wav', stop=176400) 
    if sr != 44100: continue
    y = librosa.to_mono(np.ndarray.transpose(y))
    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    y_changed = y_changed[:132300]
    ps = librosa.feature.melspectrogram(y=y_changed, sr=sr)
    if ps.shape != (128, 259): continue
    # add values to df
    df[df_idx] = [ps, row.classID]
    # print statement shows progress
    if df_idx % 200 == 0:
        print(str(df_idx)+' ('+str(round((df_idx/24500*100), 3))+'%)', end='\r')    # increment dataframe index if value added
    df_idx += 1  
joblib.dump('../Data/UrbanSound8K/aug4-df.npy', df) 

flat = df.flattened
spects_only = np.delete(df, 1, 1)
max_vals = []
for row in spects_only:
    max_vals.append(np.amax(row))

max_vals[-1:10]
# # add aug: speed 1.07
# for row in valid_data.itertuples():
#     try:
#         y, sr = librosa.load('../Data/UrbanSound8K/audio/' + row.augpathspeed1, duration=3.00, sr=44100) 
#         # if sr != 44100:
#         #     y, sr = changeSampleRate(y, sr) 
#         ps = librosa.feature.melspectrogram(y=y, sr=sr)
#         if ps.shape != (128, 259): continue
#         df.append((ps, row.classID))

#         if len(df) % 200 == 0:
#             print('Num. files added to dataset: '+str(len(df))+' ('+str(round((len(df)/len(valid_data))*100, 1))+'% complete)', end='\r')   
    
#     except IOError: continue 

# # add aug: speed 0.81
# for row in valid_data.itertuples():
#     try:
#         y, sr = librosa.load('../Data/UrbanSound8K/audio/' + row.augpathspeed2, duration=3.00, sr=44100) 
#         # if sr != 44100:
#         #     y, sr = changeSampleRate(y, sr) 
#         ps = librosa.feature.melspectrogram(y=y, sr=sr)
#         if ps.shape != (128, 259): continue
#         df.append((ps, row.classID))

#         if len(df) % 200 == 0:
#             print('Num. files added to dataset: '+str(len(df))+' ('+str(round((len(df)/len(valid_data))*100, 1))+'% complete)', end='\r')   
    
#     except IOError: continue 

# # add aug: ps 2
# for row in valid_data.itertuples():
#     try:
#         y, sr = librosa.load('../Data/UrbanSound8K/audio/' + row.augpathps1, duration=3.00, sr=44100) 
#         # if sr != 44100:
#         #     y, sr = changeSampleRate(y, sr) 
#         ps = librosa.feature.melspectrogram(y=y, sr=sr)
#         if ps.shape != (128, 259): continue
#         df.append((ps, row.classID))

#         if len(df) % 200 == 0:
#             print('Num. files added to dataset: '+str(len(df))+' ('+str(round((len(df)/len(valid_data))*100, 1))+'% complete)', end='\r')   
    
#     except IOError: continue 

# # add aug: ps 2.5
# for row in valid_data.itertuples():
#     try:
#         y, sr = librosa.load('../Data/UrbanSound8K/audio/' + row.augpathps2, duration=3.00, sr=44100) 
#         # if sr != 44100:
#         #     y, sr = changeSampleRate(y, sr) 
#         ps = librosa.feature.melspectrogram(y=y, sr=sr)
#         if ps.shape != (128, 259): continue
#         df.append((ps, row.classID))

#         if len(df) % 200 == 0:
#             print('Num. files added to dataset: '+str(len(df))+' ('+str(round((len(df)/len(valid_data))*100, 1))+'% complete)', end='\r')   
    
#     except IOError: continue 

        # # testing: ignoring filename it breaks on 
        # if file_name == '../Data/UrbanSound8K/audio/fold1/122690-6-0-0.wav' or file_name == 'fold9/180029-4-19-0.wav': continue
 