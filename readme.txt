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
									to 100, and early stopping patience to 10. This had kernels 32 and 64 in the final
									layers. 
									
Model 8							: 	Is with the cosine similarity loss function instead of the MSE. Estimates angles
									away from the axes [-1 1] and [-1 1] well, but angles at the axes extremely poor.
									For the y-axis it puts everything in the center, but at least on the axis, for the
									x axis it scatters them all over the y axis rather than keeping it at the x axis.
									
Model 9 						: 	Is exactly the same as model 7 but with the cosine similarity loss to find out
									whether the difference in score of model 8 is a consequence of the loss function 
									or a consequence of the other parameters that I changed. 
									
Model 10 						: 	I will use a combined loss function in which the cosine similarity and the MSE are
									both weighed 1 --> interrupted training too early, the model seemed to work less 
									well (i.e. the loss would go up again in the sixth epoch), but  probably if I had 
									run it for a longer time it would have worked. However, this model had 16 kernels
									in each layer only and that probably didn't help, performs better with 32 kernels 
									in the final layer
									
Model 11 						: 	Used a combined loss function but the weights were .5 for cosine similarity and 1 
									for MSE. Restored 32 kernels in final layer. This model performed well, although 
									the y-axis remains a bit of a problem. 

Model 12						: 	Same as model 11 but with 16 kernels in the final layer. Performed less well. Stopped
									training after 21 epochs. 

Model 13: 						: 	Same as model 11 but with combined loss function with weights 1 for both cosine and
									MSE. Seems to be performing similar to model 11 for now. 
									
									The main problem for all models appears to be that the y-dimension is estimated less 
									correctly. The y-dimension is mainly dependent on monaural spectral cues so it may be 
									that you are not picking up on these correctly --> In order to solve it, you could try
									a different kernel size, for instance 1x5 instead of 1x3. 
									