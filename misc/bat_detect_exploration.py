import numpy as np 
from pydub import AudioSegment 
from pydub.utils import make_chunks
import os       # getting file names from folder
import random   # this and package below used to create dummy label ...
import decimal  # ... positions in dummy files

## SEGMENTING MY AUDIO INTO 3.87 SECOND CHUNKS TO BE USED FOR TRAINING
# from https://goo.gl/WE6gjz

myaudio = AudioSegment.from_file("/home/dgabutler/CMEECourseWork/Project/Data/whinnys/5A383EF0.WAV", "wav")
chunk_length_ms = 3870
chunks = make_chunks(myaudio, chunk_length_ms)

# Export all inviduals chunks as wav files

for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    print "exporting", chunk_name
    chunk.export(chunk_name, format="wav")

# (then deleted 15th file, which is less than 3.87 seconds)

# what are the 'train files' used by run_comparison.py?
bulg_train_files = np.load(file='/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_bulgaria/train_files.npy')
# answer: a list of all 2812 file names of the train files

# 'TRAIN_FILES':
# create list of file names 
train_files = sorted(os.listdir('/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/projwav/'))
# strip file extension
train_files = [os.path.splitext(x)[0] for x in train_files]
# list to numpy array, called 'train_files' as per Mac Oadha et al. 
train_files = np.asarray(train_files)
# save array to .npy file
# NB. all np.save lines commented out as I realised np.savez was required
#np.save("/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_project/train_files", train_files)

# 'TRAIN_DURATIONS':
# first, finding out how long my clipped sections actually are:
testwav = AudioSegment.from_wav("/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/projwav/01.wav")
file_duration = testwav.duration_seconds # = 3.87 (for ms, use len(testwav))
# create 'train_durations' list, repeating length for as many files
train_durations = [file_duration] * train_files.size
# save array to .npy file
#np.save("/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_project/train_durations", train_durations)

# 'TRAIN_POS' (has to be an array of arrays):
train_pos = []
for i in range(44):
    element = np.array(([[(float(decimal.Decimal(random.randrange(0,file_duration*100000000))/100000000))]]))
    train_pos.append(element)

train_pos = np.asarray(train_pos)


# save array to .npy file
#np.save("/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_project/train_pos", train_pos)

# Create compressed npz file from three dummy arrays
np.savez_compressed('/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_project',train_pos=train_pos, train_durations=train_durations, train_files=train_files)

##### ATTEMPT AT TRAINING USING THE DUMMY DATA

cd /home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train

# (most copied from run_comparison.py)
import numpy as np
import matplotlib.pyplot as plt
import os
import sys # added 04.07.18
sys.path.insert(0, '/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train')
import evaluate as evl
import create_results as res
from data_set_params import DataSetParams
import classifier as clss
import pandas as pd
import cPickle as pickle

test_set = 'project'    # if 'project', my data. if 'uk', their data
data_set = '/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_' + test_set + '.npz'
raw_audio_dir = '/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/projwav/'
base_line_dir = '/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/baselines/'
result_dir = '/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/results/'
model_dir = '/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/models/'
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
print 'test set:', test_set
plt.close('all')

# train and test_pos are in units of seconds
loaded_data_tr = np.load(data_set)
train_pos = loaded_data_tr['train_pos']
train_files = loaded_data_tr['train_files']
train_durations = loaded_data_tr['train_durations']
# test_pos = loaded_data_tr['test_pos']
# test_files = loaded_data_tr['test_files']
# test_durations = loaded_data_tr['test_durations']

# load parameters
params = DataSetParams()
params.audio_dir = raw_audio_dir

#
# CNN
print '\ncnn'
params.classification_model = 'cnn'
model = clss.Classifier(params)
# train and test
model.train(train_files, train_pos, train_durations)
nms_pos, nms_prob = model.test_batch(test_files, test_pos, test_durations, False, '')
# compute precision recall
precision, recall = evl.prec_recall_1d(nms_pos, nms_prob, test_pos, test_durations, model.params.detection_overlap, model.params.window_size)
res.plot_prec_recall('cnn', recall, precision, nms_prob)
# save CNN model to file
pickle.dump(model, open(model_dir + 'test_set_' + test_set + '.mod', 'wb'))



## The following is messing around to see what the 
## position/duration'.npy' files were
test_durations = np.load(file='/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_norfolk/test_durations.npy')
test_positions = np.load(file='/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_norfolk/test_pos.npy')
train_durations = np.load(file='/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_norfolk/train_durations.npy')
norf_train_positions = np.load(file='/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_norfolk/train_pos.npy')
bulg_train_positions = np.load(file='/home/dgabutler/CMEECourseWork/Project/Sandbox/batdetect/bat_train/data/train_test_split/test_set_bulgaria/train_pos.npy')
# Saving unweildy numpy array to text file for closer inspection
# ... 'train-positions':
# norfolk
np.savetxt("norf-train-positions.txt", norf_train_positions, fmt='%5s',delimiter=',')
# bulgaria
np.savetxt("bulg-train-positions.txt", bulg_train_positions, fmt='%5s',delimiter=',')
# ... 'test-positions'
np.savetxt("test-positions.txt", test_positions, fmt='%5s',delimiter=',')