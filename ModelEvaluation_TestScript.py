# script to analyze the model

# set directories
dirfiles = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\DNN_ESANN'
dirscripts = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\DNN_ESANN\DNN_ESANN_dev'

# import required packages and libraries
from tensorflow.keras.models import load_model
import os
import numpy as np

os.chdir(dirscripts)
from CustLoss_MSE import cust_mean_squared_error

# load model
model = load_model(dirfiles+'/model3.h5', custom_objects={"cust_mean_squared_error": cust_mean_squared_error})

# load history of the model?


# load the data
an_l_test = np.load(dirfiles+"/an_l_test.npy")
an_r_test = np.load(dirfiles+"/an_r_test.npy") 
labels_test =  np.load(dirfiles+"/labels_test.npy")

# expand dimensions for the model
an_l_test = np.expand_dims(an_l_test,axis = 3)
an_r_test = np.expand_dims(an_r_test,axis = 3)

# prepare data for model evaluation
X_test = [an_l_test, an_r_test]
Y_test = labels_test

# evaluate the model
score = model.evaluate(X_test, Y_test, verbose=1)

