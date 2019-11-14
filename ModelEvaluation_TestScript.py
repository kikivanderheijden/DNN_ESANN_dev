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
import pickle

os.chdir(dirscripts)
from CustLoss_MSE import cust_mean_squared_error
from ModelPredictions import generate_model_predictions

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
hist = pandas.read_csv(dirfiles+"/history_"+modelname+".csv")

# load weights of all layers
modelweights = model.get_weights()

# load the data
an_l_test = np.load(dirfiles+"/an_l_test_18000_"+modelname+".npy")
an_r_test = np.load(dirfiles+"/an_r_test_18000_"+modelname+".npy") 
labels_test =  np.load(dirfiles+"/labels_test_18000_"+modelname+".npy")
names_test = pickle.load(open(dirfiles+'/listfilenames_18000_test_'+modelname+'.p','rb'))

# prepare data for model evaluation
X_test = [an_l_test, an_r_test]
Y_test = labels_test

#------------------------------------------------------------------------------
# Run model evaluation
#------------------------------------------------------------------------------

# predict --> create predictions (set to 1) or load predictions (set to 0)
newpredictions = 0
if newpredictions == 1:
    predictions = generate_model_predictions(model, X_test, modelname, dirfiles)
elif newpredictions == 0:
    predictions = np.load(dirfiles+"/predictions_"+modelname+".npy")
        
# evaluate the model on unseen data (shouldn't be test data)
score = model.evaluate(X_test, Y_test, verbose=1)

#------------------------------------------------------------------------------
# Analyze predictions
#------------------------------------------------------------------------------

# retrieve test labels from names 
names_test_angle = np.empty([len(names_test)])
cnt1 = 0
for x in names_test:
    names_test_angle[cnt1] = x[1:4]
    cnt1 += 1
    
# compute angular error
    
#Kiki's method --> not correct because at the border it computes the error without
    #wrap around
# convert predicted angles to degrees
predangles = np.empty([len(predictions)])
for x in range(len(predictions)):
    tempx = predictions[x,0]
    tempy = predictions[x,1]
    if tempx >=0 and tempy >= 0:
        preddeg = np.arcsin(predictions[x,0]/np.sqrt(np.square(predictions[x,0])+np.square(predictions[x,1])))*180/np.pi
    elif tempx >=0 and tempy < 0:
        preddeg = 180-np.arcsin(predictions[x,0]/np.sqrt(np.square(predictions[x,0])+np.square(predictions[x,1])))*180/np.pi
    elif tempx< 0 and tempy < 0:
        preddeg = 180+np.abs(np.arcsin(predictions[x,0]/np.sqrt(np.square(predictions[x,0])+np.square(predictions[x,1]))))*180/np.pi
    elif tempx< 0 and tempy >=0:
        preddeg = 360+np.arcsin(predictions[x,0]/np.sqrt(np.square(predictions[x,0])+np.square(predictions[x,1])))*180/np.pi
    predangles[x] = preddeg
#compute difference between predicted angles and ground truth angles, correct for difference > 180 deg, that's not possible
angular_error_kiki = np.empty([len(predangles)])
for  x in range(len(predangles)):
    tempangularerror = predangles[x]-names_test_angle[x]
    if np.abs(tempangularerror) > 180:
        tempangularerror = np.abs(360-np.abs(tempangularerror))
    angular_error_kiki[x] = tempangularerror
    
    
plt.figure()
plt.plot(angular_error_kiki)    

predangles[19]
names_test_angle[19]

# Adavanne's method
angular_error_ada = np.empty([len(predictions)])
for x in range(len(predictions)):
    xpred = predictions[x,0]
    ypred = predictions[x,1]
    xreal = labels_test[x,0]
    yreal = labels_test[x,1]
    # testvals[x] = np.around(np.sqrt(np.square(xpred-xreal)+np.square(ypred-yreal))/2,1)
    # first compute the mse
    MSE_temp = np.sqrt(np.square(xpred-xreal)+np.square(ypred-yreal))/2
    if MSE_temp > 1:
        MSE_temp = np.around(MSE_temp,1) # round the number as np.arcsin() does not work on numbers > 1
    angular_error_ada[x] = 2*np.arcsin(MSE_temp)*(180/np.pi)
    
plt.figure()
plt.plot(angular_error_ada)

plt.figure()
plt.scatter([xpred,xreal],[ypred,yreal])
plt.axis('square')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)

predangles[19]
names_test_angle[19]
    
#------------------------------------------------------------------------------
# Create figures
#------------------------------------------------------------------------------    
# to create a histogram
plt.figure()
plt.hist(names_test_angle)
    
# to create scatterplots
anglecheck1 = 0
color1 = (1,0,0)
anglecheck2 = 90
color2 = (0.98, 0.95, 0.35)
anglecheck3 = 180
color3 = (0.62,1,0.24)
anglecheck4 = 270
color4 = (0.35,0.5,0.98)
plt.figure()
plt.scatter(predictions[np.squeeze(names_test_angle==anglecheck1),0],predictions[np.squeeze(names_test_angle==anglecheck1),1],color=color1, alpha=0.4)
plt.scatter(labels_test[np.squeeze(names_test_angle==anglecheck1),0],labels_test[np.squeeze(names_test_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_test_angle==anglecheck2),0],predictions[np.squeeze(names_test_angle==anglecheck2),1],color=color2, alpha=0.4)
plt.scatter(labels_test[np.squeeze(names_test_angle==anglecheck2),0],labels_test[np.squeeze(names_test_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_test_angle==anglecheck3),0],predictions[np.squeeze(names_test_angle==anglecheck3),1],color=color3, alpha=0.4)
plt.scatter(labels_test[np.squeeze(names_test_angle==anglecheck3),0],labels_test[np.squeeze(names_test_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_test_angle==anglecheck4),0],predictions[np.squeeze(names_test_angle==anglecheck4),1],color=color4, alpha=0.4)
plt.scatter(labels_test[np.squeeze(names_test_angle==anglecheck4),0],labels_test[np.squeeze(names_test_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.axis('square')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)

anglecheck1 = 40
color1 = (147/255,248/255,254/255)
anglecheck2 = 140
color2 = (36/255,31/255,249/255)
anglecheck3 = 220
color3 = (218/255,43/255,200/255)
anglecheck4 = 320
color4 = (1,0,0)
plt.figure()
plt.scatter(predictions[np.squeeze(names_test_angle==anglecheck1),0],predictions[np.squeeze(names_test_angle==anglecheck1),1],color=color1, alpha=0.4)
plt.scatter(labels_test[np.squeeze(names_test_angle==anglecheck1),0],labels_test[np.squeeze(names_test_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_test_angle==anglecheck2),0],predictions[np.squeeze(names_test_angle==anglecheck2),1],color=color2, alpha=0.4)
plt.scatter(labels_test[np.squeeze(names_test_angle==anglecheck2),0],labels_test[np.squeeze(names_test_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_test_angle==anglecheck3),0],predictions[np.squeeze(names_test_angle==anglecheck3),1],color=color3, alpha=0.4)
plt.scatter(labels_test[np.squeeze(names_test_angle==anglecheck3),0],labels_test[np.squeeze(names_test_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_test_angle==anglecheck4),0],predictions[np.squeeze(names_test_angle==anglecheck4),1],color=color4, alpha=0.4)
plt.scatter(labels_test[np.squeeze(names_test_angle==anglecheck4),0],labels_test[np.squeeze(names_test_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.axis('square')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)

preddeg = np.arcsin(predictions[0,1]/np.sqrt(np.square(predictions[0,0])+np.square(predictions[0,1])))*360/np.pi
preddeg = np.arcsin(np.around(np.sqrt(np.square(labels_test[0,0])+np.square(labels_test[0,1])),decimals=2))*360/np.pi

# convert predicted angles to degrees
predangles = np.empty([len(predictions)])
for x in range(len(predictions)):
    tempx = predictions[x,0]
    tempy = predictions[x,1]
    if tempx >=0 and tempy >= 0:
        preddeg = np.arcsin(predictions[x,0]/np.sqrt(np.square(predictions[x,0])+np.square(predictions[x,1])))*180/np.pi
    elif tempx >=0 and tempy < 0:
        preddeg = 180-np.arcsin(predictions[x,0]/np.sqrt(np.square(predictions[x,0])+np.square(predictions[x,1])))*180/np.pi
    elif tempx< 0 and tempy < 0:
        preddeg = 180+np.abs(np.arcsin(predictions[x,0]/np.sqrt(np.square(predictions[x,0])+np.square(predictions[x,1]))))*180/np.pi
    elif tempx< 0 and tempy >=0:
        preddeg = 360+np.arcsin(predictions[x,0]/np.sqrt(np.square(predictions[x,0])+np.square(predictions[x,1])))*180/np.pi
    predangles[x] = preddeg

# create figure of location and   
plt.figure()
plt.plot(names_test_angle[names_test_angle==90])
plt.plot(predangles[np.where(names_test_angle==90)[0]])

anglecheck = 20

theta = predangles[np.where(names_test_angle==anglecheck)[0]]
r = np.ones(len(predangles[np.where(names_test_angle==anglecheck)[0]]))
    
fig = plt.figure()
ax = fig.add_subplot(111,projection = 'polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
c = ax.scatter(np.radians(theta), r)



# creating a plot of the loss (i.e. model performance)

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
