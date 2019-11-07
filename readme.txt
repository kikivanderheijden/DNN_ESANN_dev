File structure DNN_ESANN_dev

------------------------------------------------------------------------------------------
Relevant files
------------------------------------------------------------------------------------------
LoadAndProcess_Sounds.py 		:   This script loads the auditory nerve representation of the sounds
									and creates three numpy arrays: trainlabels, an_l (left channel of 
									auditory nerve representation), an_r (right channel of auditory 
									nerve representation. Numpy arrays are saved to disk.
									
ImportAndPrepare_Data.py 		:	This script is a function that prepares data for deep learning pipeline. 
									The numpy arrays containing the relevant representations are loaded and 
									divided in a train and test split. 

CreateDNN_architecture.py 		: 	This script creates the DNN model, prints a summary, and saves the 
									model as an .h5 file. 

TrainDNN.py 					: 	This script loads the data (using the function ImportAndPrepare_Data.py) 
									and the model and trains the DNN. 								
