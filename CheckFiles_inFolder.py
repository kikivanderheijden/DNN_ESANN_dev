# script to check files in folder

# set directories
dir_anfiles = "/home/jovyan/Data/TestCochSoundsForDNN" # for DSRI
#dir_anfiles = r"C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning\Sounds\TestCochSoundsForDNN_small" # for local testing

# import necessary packages and libraries
import os # to get info about directories
import numpy as np
import math
from pytictoc import TicToc 
t = TicToc() # create instant of class
from scipy.io import loadmat

# set parameters
time_sound = 2000 # time dimension of sound files, i.e. number of samples
nfreqs     = 99 # nr of freqs used

# start script
t.tic()
countfiles = 0
with os.scandir(dir_anfiles) as listfiles:
    for entry in listfiles:
        countfiles = countfiles+1
        #print(entry.name) # use this to display all file names
        #print(temploc) 
        #print(trainlabels)  
t.toc("scanning files took")
print("this folder contains" , countfiles, "files" )

t.tic()
 # create array of location labels for the x and y coordinates. labels range from -1 to 1 in correspondence with the unit circle 
# take labels from directory for left channel but they are the same for the left and right channel
trainlabels_x = [] # initialize array of labels
trainlabels_y = []
with os.scandir(dir_anfiles) as listfiles:
    for entry in listfiles:
        # derive location from filename 
        temploc = int(entry.name[1:4]) # get azimuth location and convert to integer
        # note that the naming of the files has 0 at front, while for the unit circle 0 should be at the right, correct this first
        if temploc >= 0 and temploc <= 90:
            temploc = np.abs(temploc - 90)
        elif temploc > 90:
            temploc = np.abs(temploc - 90- 360)
        temp_xcoord = math.cos(math.radians(temploc)) # math.cos operates on radians so convert angle to rad first
        temp_ycoord = math.sin(math.radians(temploc))
        trainlabels_x.append(temp_xcoord)
        trainlabels_y.append(temp_ycoord)
        #print(entry.name) # use this to display all file names
        #print(temploc) 
        #print(trainlabels)   
# add together in 2D array where column 1 = x coord and column 2 = y coord
trainlabels = np.vstack((np.array(trainlabels_x),np.array(trainlabels_y)))
trainlabels = np.transpose(trainlabels)

t.toc("creating the train labels took")
print("shape of training labels is ", trainlabels.shape)    

t.tic()
countfilesdone = 0
countfiles100 = 0
# find and read files
train_an_l = []
train_an_r = []
#train_an_l = np.empty([1,time_sound,nfreqs]) # note that in a 3d array, the first dimension specificies the matrix, the second row, 
                                           # and the third column, and remember that all indices start at 0!!!!
#train_an_r = np.empty([1,time_sound,nfreqs]) # note that in a 3d array, the first dimension specificies the matrix, the second row, 
                                           # and the third column, and remember that all indices start at 0!!!!
with os.scandir(dir_anfiles) as listfiles:
    for entry in listfiles:
        tempdata_l = loadmat(dir_anfiles+"/"+entry.name)['AN_l']
        tempdata_r = loadmat(dir_anfiles+"/"+entry.name)['AN_r']
        train_an_l.append(tempdata_l)
        train_an_r.append(tempdata_r)
#        tempdata_l = np.atleast_3d(tempdata_l) # convert into 3D matrix 
#        tempdata_r = np.atleast_3d(tempdata_r) # convert into 3D matrix 
#        tempdata_l = np.reshape(tempdata_l,(1,time_sound,nfreqs)) # reshape into correct dimensions
#        tempdata_r = np.reshape(tempdata_r,(1,time_sound,nfreqs)) # reshape into correct dimensions
#        train_an_l = np.append(train_an_l,tempdata_l,axis = 0) # when arrays are same dimension, append along first dimension     
#        train_an_r = np.append(train_an_r,tempdata_r,axis = 0) # when arrays are same dimension, append along first dimension   
                    #print(entry_r.name)
#        countfilesdone = countfilesdone+1
        if countfilesdone == 100:
            print("100 files done")
            countfiles100 = countfiles100+1
            t.toc("these 100 files took")
            t.toc(restart=True)
            countfilesdone = 0
#train_an_l = train_an_l[1:] # delete first matrix which was used to initialize, keep all others
#train_an_r = train_an_r[1:] # delete first matrix which was used to initialize, keep all others
t.toc("loading the train sounds took ")
print("shape of training sounds is", len(train_an_l))

train_an_l_array = np.asarray(train_an_l)
train_an_r_array = np.asarray(train_an_r)
print("shape of the training sounds array is ", train_an_l_array.shape)
