# script to check files in folder

# set directories
dir_anfiles = "/home/jovyan/Data/TestCochSoundsForDNN" # sounds left channel

# import necessary packages and libraries
import os # to get info about directories
import numpy as np
import math
from pytictoc import TicToc
t = TicToc() # create instant of class

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
print("shape of training sounds is ", trainlabels.shape)    
