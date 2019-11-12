# specify directories
dir_wrfiles = "/home/jovyan/DNN_ESANN_dev" # for testing on DSRI
#dir_wrfiles = r"C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning"
# for testing locally

# script to create DNN architecture
from tensorflow.keras import layers
from tensorflow.keras import models # contains different types of models (use sequential model here?)
from tensorflow.keras import optimizers # contains different types of back propagation algorithms to train the model, 
                                        # including sgd (stochastic gradient

#import os
#os.chdir(r"C:\Users\kiki.vanderheijden\Documents\PYTHON\DNN_ESANN\DNN_ESANN_dev")
from CustLoss_MSE import cust_mean_squared_error # note that in this loss function, the axis of the MSE is set to 1


# specifications of the input
time_sound = 2000
nfreqs = 99

# model left channel
in1 = layers.Input(shape=(time_sound,nfreqs,1)) # define input (rows, columns, channels (only one in my case))
model_l_conv1 = layers.Conv2D(32,(1,3),activation='relu', padding = 'same')(in1) # define first layer and input to the layer
model_l_conv1_mp = layers.MaxPooling2D(pool_size = (1,3))(model_l_conv1)
model_l_conv1_mp_do = layers.Dropout(0.2)(model_l_conv1_mp)

# model right channel
in2 = layers.Input(shape=(time_sound,nfreqs,1)) # define input
model_r_conv1 = layers.Conv2D(32,(1,3),activation='relu', padding = 'same')(in2) # define first layer and input to the layer
model_r_conv1_mp = layers.MaxPooling2D(pool_size = (1,3))(model_r_conv1)
model_r_conv1_mp_do = layers.Dropout(0.2)(model_r_conv1_mp)

# model channels merged
model_final_merge = layers.Subtract()([model_l_conv1_mp_do, model_r_conv1_mp_do]) # does the number of parameters change if you do not make this into a subtraction
                                                                      # but use it as two channels instead? 
model_final_conv1 = layers.Conv2D(64,(3,3),activation='relu', padding = 'same')(model_final_merge)
model_final_conv1_mp = layers.MaxPooling2D(pool_size = (2,3))(model_final_conv1)
model_final_conv1_mp_do = layers.Dropout(0.2)(model_final_conv1_mp)
#model_final_conv2 = layers.Conv2D(128,(1,3),activation='relu', padding = 'same')(model_final_conv1_mp)
#model_final_conv2_mp = layers.MaxPooling2D(pool_size = (1,4))(model_final_conv2)

model_final_flatten = layers.Flatten()(model_final_conv1_mp_do)
model_final_dropout = layers.Dropout(0.2)(model_final_flatten) # dropout for regularization
predicted_coords = layers.Dense(2, activation = 'tanh')(model_final_dropout) # I have used the tanh activation because our outputs should be between -1 and 1

# create model
model = models.Model(inputs = [in1,in2], outputs = predicted_coords)

# compile model
model.compile(loss = cust_mean_squared_error, optimizer = optimizers.Adam(), metrics=['mse'])

model.summary()
model.save(dir_wrfiles+'/DNN_model5.h5') # save model

