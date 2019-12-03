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

Model 13 						: 	Same as model 11 but with combined loss function with weights 1 for both cosine and
									MSE. Seems to be performing similar to model 11 for now. Was slightly worse than
									model 11. 
									
Model 14 						: 	Was stupid, wanted to use a larger kernel size than model 11 (otherwise the same model)
									but set one dimension to an even number, i.e. 2x5. So results of this model should be
									disregarded. 

Model 15 						: 	 Same as model 14 but with kernel size 3x5 in first layer. 
									
									
--------------------------------------------------------------------------------------------
Formal model evaluation
--------------------------------------------------------------------------------------------
* Include the following models in the formal evaluation: 

	Model 11: 		MSE & cos, 1 & .5

	Model 13: 		MSE & cos, 1 & 1
								
	Model 15: 		MSE & cos, 1 & .5, larger kernel size	

	Model 16: 		MSE only

	Model 17: 		cos only
	
	Model 18:		angular distance instead of cosine distance
	
	Model 19: 		Same as model 18, redone because model 18 crashed --> timing start Mon 2019_12_02 11:43