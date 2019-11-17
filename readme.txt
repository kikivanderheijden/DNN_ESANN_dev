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

									

-------------------------------------------------------------------------------------------
Pipeline
-------------------------------------------------------------------------------------------

[0] optional - sound preparation:	[LoadAndProcess_Sounds.py] This script needs to be executed only if new 
									sounds have been added or if the sounds have been changed. 
									
[0] optional - create model 	: 	[CreateDNN_architecture_modelX.py] This script needs to be executed only 			
									when a new model should be created. 
									
[1] training - perform DNN 		:   [TrainDNN.py] Run this script to train the model. Make sure that the 
									correct set of sounds + model is loaded, processed, and saved. 

--------------------------------------------------------------------------------------------
Model versions
--------------------------------------------------------------------------------------------

Model 4 						: 	Describe details 

Model 5							: 	Added Dropout = 0.2 to every layer of the model; batch size = 64

Model 6							: 	Replaced merging-subtract with merging-concatenate, added early stopping
									--> note that you set the 'Restore best weights' to False, maybe it's better
									to set that to True? Check model history callback to see if you can retrieve 
									the weights of the best model. 

Model 7							: 	This is the same model as model 6, but with batch size 128, number of epochs 
									to 100, and early stopping patience to 10
									
Model 8							: 	Is with the cosine similarity loss function instead of the MSE. Estimates angles
									away from the axes [-1 1] and [-1 1] well, but angles at the axes extremely poor.
									For the y-axis it puts everything in the center, but at least on the axis, for the
									x axis it scatters them all over the y axis rather than keeping it at the x axis.
									
Model 9 						: 	Is exactly the same as model 7 but with the cosine similarity loss to find out
									whether the difference in score of model 8 is a consequence of the loss function 
									or a consequence of the other parameters that I changed. 
									
Model 10 						: 	I will use a combined loss function in which the cosine similarity and the MSE are
									both weighed 1.