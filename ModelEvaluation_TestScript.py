# script to analyze model performance

#------------------------------------------------------------------------------
# Specifications
#------------------------------------------------------------------------------

# set directories
dirfiles = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\DNN_ESANN'
dirscripts = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\DNN_ESANN\DNN_ESANN_dev'

# import required packages and libraries
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

os.chdir(dirscripts)
from CustLoss_MSE import cust_mean_squared_error

# define name of current model
modelname = "model5"

# model parameters for evaluation
sizebatches = 64

#------------------------------------------------------------------------------
# Preparations
#------------------------------------------------------------------------------
# load model
model = load_model(dirfiles+'/'+modelname+'_final.h5', custom_objects={"cust_mean_squared_error": cust_mean_squared_error})

# load history of the model
hist = pandas.read_csv(dirfiles+"/history_model5.csv")

# load weights of all layers
modelweights = model.get_weights()

# load the data
an_l_test = np.load(dirfiles+"/an_l_test_18000_"+modelname+".npy")
an_r_test = np.load(dirfiles+"/an_r_test_18000_"+modelname+".npy") 
labels_test =  np.load(dirfiles+"/labels_test_18000_"+modelname+".npy")

# prepare data for model evaluation
X_test = [an_l_test, an_r_test]
Y_test = labels_test

#------------------------------------------------------------------------------
# Model evaluation
#------------------------------------------------------------------------------

# evaluate
score = model.evaluate(X_test, Y_test, verbose=1)
predictions = model.predict(X_test, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=10)

print(predictions[:10,])
print(labels_test[:10,])

#meanlabel = np.mean(Y_test, axis = 0)
#print(meanlabel)


#labels_train = np.load(dirfiles+"/labels_train.npy")
#meanlabeltrain = np.mean(labels_train, axis = 0)
#print(meanlabeltrain)

plt.figure()
plt.plot(hist.loss)
plt.plot(hist.val_loss)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

plt.figure()
plt.plot(hist.mean_squared_error)
plt.plot(hist.val_mean_squared_error)
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
