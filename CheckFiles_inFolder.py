# script to check files in folder

# set directories
dir_anfiles = "/home/jovyan/Data/TestCochSoundsForDNN" # sounds left channel

# import necessary packages and libraries
import os # to get info about directories

countfiles = 0
with os.scandir(dir_anfiles) as listfiles:
    for entry in listfiles:
        countfiles = countfiles+1
        #print(entry.name) # use this to display all file names
        #print(temploc) 
        #print(trainlabels)  

print("this folder contains" , countfiles, "files" )