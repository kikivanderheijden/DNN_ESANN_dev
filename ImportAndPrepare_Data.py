
# read and preprocess sounds in the DSRI environment

# define directories, add r before the name because a normal string cannot be used as a path, alternatives are 
# using / or \\ instead of \
dir_anfiles = "/home/jovyan/Data/TestCochSoundsForDNN" # sounds left channel

# import packages and libraries
import os # to get info about directories
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import gc # garbage collector

# import functions to call them later
os.chdir("/home/jovyan/DNN_ESANN")
import read_and_process_audmat

# define parameters
time_sound = 2000 # time dimension of sound files, i.e. number of samples
nfreqs     = 99 # nr of freqs used

# returns trainlabels (2d array with first column = x coord and 2nd column = y coord, and AN representation left and right)
labels, an_l, an_r = read_and_process_audmat.audmat(dir_anfiles, time_sound, nfreqs)

# shuffle data in the same way
labels_rand, an_l_rand, an_r_rand = shuffle(labels, an_l, an_r, random_state = 0)

# if shuffling is  OK, remove unused variables from memory
del labels
del an_l
del an_r
gc.collect() # collect garbage to save memory

# create a train and test split
labels_rand_train, labels_rand_test, an_l_rand_train, an_l_rand_test, an_r_rand_train, an_r_rand_test = train_test_split(labels_rand, an_l_rand, an_r_rand, test_size = 0.15, shuffle = False)

# clean memory
del labels_rand
del an_l_rand
del an_r_rand
gc.collect()

print("Shape of training sounds is:", an_l_rand_train.shape)
print("Shape of training sounds is:", an_r_rand_train.shape)
print("Shape of training labels is:", labels_rand_train.shape)
