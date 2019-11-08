# Main to train a DNN model

dir_fufiles = "/home/jovyan/DNN_ESANN_dev" # specify directory where function files are located
dir_mofiles = "/home/jovyan/DNN_ESANN_dev" # specify directory where model files are located 

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

# specifications of the model                                        
batch_size = 32 # a parameter of gradient descent that controls the number of training samples to work through 
                # before the model's internal parameters are updated
epoch_nr = 100 # number of epochs to train the model, an epoch is an iteration over the entire x and y data
#adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) # optimizer with default settings

# model left channel
in1 = layers.Input(shape=(time_sound,nfreqs,1)) # define input (rows, columns, channels (only one in my case))
model_l_conv1 = layers.Conv2D(32,(1,3),activation='relu', padding = 'same')(in1) # define first layer and input to the layer
model_l_conv1_bn = layers.BatchNormalization()(model_l_conv1)

# model right channel
in2 = layers.Input(shape=(time_sound,nfreqs,1)) # define input
model_r_conv1 = layers.Conv2D(32,(1,3),activation='relu', padding = 'same')(in2) # define first layer and input to the layer
model_r_conv1_bn = layers.BatchNormalization()(model_r_conv1)

# model channels merged
model_final_merge = layers.Subtract()([model_l_conv1_bn, model_r_conv1_bn]) # does the number of parameters change if you do not make this into a subtraction
                                                                      # but use it as two channels instead? 
model_final_conv1 = layers.Conv2D(64,(1,3),activation='relu', padding = 'same')(model_final_merge)
model_final_conv1_bn = layers.BatchNormalization()(model_final_conv1) # I'm adding the batch normalization after the 
                                                                      # activation function, but it may also be good to do
                                                                      # to do it before the activation
model_final_conv1_mp = layers.MaxPooling2D(1,2)(model_final_conv1_bn)

model_final_conv2 = layers.Conv2D(128,(1,3),activation='relu', padding = 'same')(model_final_conv1_mp)
model_final_conv2_bn = layers.BatchNormalization()(model_final_conv2) # I'm adding the batch normalization after the 
                                                                      # activation function, but it may also be good to do
                                                                      # to do it before the activation
model_final_conv2_mp = layers.MaxPooling2D(1,2)(model_final_conv2_bn)

model_final_flatten = layers.Flatten()(model_final_conv2_mp)
model_final_dropout = layers.Dropout(0.2)(model_final_flatten) # dropout for regularization
#model_final_flatdense = layers.Dense(100, activation = 'relu')(model_final_dropout)
predicted_coords = layers.Dense(2, activation = 'linear')(model_final_dropout)

# create model
model = models.Model(inputs = [in1,in2], outputs = predicted_coords)

# compile model
model.compile(loss = cust_mean_squared_error, optimizer = optimizers.Adam(), metrics=['mse'])

model.summary()
model.save(dir_wrfiles+'/DNN_model2.h5') # save model

#######################################
#PART 2 training the DNN
#######################################

#dir_mofiles = r"C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning"

# import packages, libraries, functions to call them later
import ImportAndPrepare_Data
from pytictoc import TicToc
t = TicToc()
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform

# load data
labels_rand_train, labels_rand_test, an_l_rand_train, an_l_rand_test, an_r_rand_train, an_r_rand_test = ImportAndPrepare_Data.im_and_prep()

# load model
t.tic()
mymodel = load_model(dir_mofiles+"/DNN_model2.h5",custom_objects={'GlorotUniform': glorot_uniform()})
mymodel.summary()
t.toc("loading the model took ")


# train the model
t.tic()
history = mymodel.fit([an_l_rand_train, an_r_rand_train], labels_rand_train, validation_data=((an_l_rand_test,an_r_rand_test),labels_rand_test), epochs = 100, batch_size = 32, use_multiprocessing = True)
#.toc("training the model took ")

# =============================================================================
#from tensorflow.keras.models import model_from_json
# t.tic()
# json_file = open(dir_mofiles+'/DNN_model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.summary()
# t.toc("loading the model took")
# =============================================================================
