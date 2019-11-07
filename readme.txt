File structure DNN_ESANN_dev

------------------------------------------------------------------------------------------
Relevant files
------------------------------------------------------------------------------------------
LoadAndProcess_Sounds.py 		:   This script loads the auditory nerve representation of the sounds
									and creates three numpy arrays: trainlabels, an_l (left channel of 
									auditory nerve representation), an_r (right channel of auditory 
									nerve representation. Numpy arrays are saved to disk.
									
ImportAndPrepare_Data.py 		:	This script prepares data for deep learning pipeline. 