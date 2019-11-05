# function to read and process .csv files with AN responses

def audmat(dir_anfiles, time_sound, nfreqs):
    """
    This function reads the csv files and returns the labels in x,y coordinates of the unit circle, as well as the sounds in numpy arrays.
    """
    
    # import packages/libraries
    import os
    import numpy as np
    import math
    from scipy.io import loadmat
      
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
    
    # find and read files
    train_an_l = np.empty([1,time_sound,nfreqs]) # note that in a 3d array, the first dimension specificies the matrix, the second row, 
                                               # and the third column, and remember that all indices start at 0!!!!
    train_an_r = np.empty([1,time_sound,nfreqs]) # note that in a 3d array, the first dimension specificies the matrix, the second row, 
                                               # and the third column, and remember that all indices start at 0!!!!
    with os.scandir(dir_anfiles) as listfiles:
        for entry in listfiles:
            tempdata_l = loadmat(dir_anfiles+"/"+entry.name)['AN_l']
            tempdata_r = loadmat(dir_anfiles+"/"+entry.name)['AN_r']
            tempdata_l = np.atleast_3d(tempdata_l) # convert into 3D matrix 
            tempdata_r = np.atleast_3d(tempdata_r) # convert into 3D matrix 
            tempdata_l = np.reshape(tempdata_l,(1,time_sound,nfreqs)) # reshape into correct dimensions
            tempdata_r = np.reshape(tempdata_r,(1,time_sound,nfreqs)) # reshape into correct dimensions
            train_an_l = np.append(train_an_l,tempdata_l,axis = 0) # when arrays are same dimension, append along first dimension     
            train_an_r = np.append(train_an_r,tempdata_r,axis = 0) # when arrays are same dimension, append along first dimension   
                        #print(entry_r.name)
    train_an_l = train_an_l[1:] # delete first matrix which was used to initialize, keep all others
    train_an_r = train_an_r[1:] # delete first matrix which was used to initialize, keep all others

    return trainlabels, train_an_l, train_an_r   

    

