# script to analyze model performance

# to clear all variables: type in ipython console %reset

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
from CustLoss_Combined_Cosine_MSE_weighed import cos_dist_2D_and_mse_weighed # note that in this loss function, the axis of the MSE is set to 1
from CustMet_cosine_distance import cos_distmet_2D



# define name of current model
modelname = "model11"

# model parameters for evaluation
sizebatches = 128

# set azimuth range
azimuthrange = np.arange(0,360,10)

#------------------------------------------------------------------------------
# Preparations
#------------------------------------------------------------------------------
# load model
#model = load_model(dirfiles+'/'+modelname+'.h5', custom_objects={"cust_mean_squared_error": cust_mean_squared_error, "cos_dist_2D_and_mse_weighed": cos_dist_2D_and_mse_weighed, "cos_distmet_2D": cos_distmet_2D})
model = load_model(dirfiles+'/'+modelname+'_final.h5', custom_objects={"cust_mean_squared_error": cust_mean_squared_error, "cos_dist_2D_and_mse_weighed": cos_dist_2D_and_mse_weighed, "cos_distmet_2D": cos_distmet_2D})

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
newpredictions = 1
if newpredictions == 1:
    predictions = generate_model_predictions(model, X_test, modelname, dirfiles, sizebatches)
elif newpredictions == 0:
    predictions = np.load(dirfiles+"/predictions_"+modelname+".npy")
        
# evaluate the model on unseen data (shouldn't be test data)
# score = model.evaluate(X_test, Y_test, verbose=1)

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
# Kiki's method 
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
    tempangularerror = np.abs(predangles[x]-names_test_angle[x])
    if tempangularerror > 180:
        tempangularerror = np.abs(360-np.abs(tempangularerror))
    angular_error_kiki[x] = tempangularerror

cosine_distance_degrees = np.empty([len(predangles)])
for  x in range(len(predangles)):
     cossim = np.sum(labels_test[x]*predictions[x])/(np.sqrt(np.sum(np.square(labels_test[x])))*np.sqrt(np.sum(np.square(predictions[x]))))
     cosine_distance_degrees[x] = np.arccos(cossim)*180/np.pi
     
# compute mean and standard deviation of error per target azimuth location
mean_angular_error_pertargetangle = np.empty([len(azimuthrange)]) 
stdev_angular_error_pertagetanle = np.empty([len(azimuthrange)]) 
angular_error_pertargetangle = [None] *len(azimuthrange)
for x in range(len(azimuthrange)):
    mean_angular_error_pertargetangle[x] = np.mean(angular_error_kiki[np.where(names_test_angle==azimuthrange[x])])
    stdev_angular_error_pertagetanle[x] = np.std(angular_error_kiki[names_test_angle==azimuthrange[x]])
    angular_error_pertargetangle[x] = np.ndarray.tolist(angular_error_kiki[names_test_angle==azimuthrange[x]])

# create boxplot for all targetazimuths
# change order of azms such that 0 deg is in the middle
data = angular_error_pertargetangle[-18:]+angular_error_pertargetangle[:18]
plt.figure()
flierprops = dict(marker='+', markerfacecolor='black', alpha = .5, markersize=3,
                  linestyle='none')
plt.boxplot(data, flierprops=flierprops)
ind = np.arange(1,37,2)
plt.xticks(ind,('180','200','220','240','260','280','300','320','340','0','20','40','60','80','100','120','140','160'),rotation=90)

#plt.xticks(ind,('180','190','200','210','220','230','240','250','260','270','280','290','300','310','320','330','340','350','0','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150','160','170'))
plt.xlabel('Target Locations (degrees)')
plt.ylabel('Error (degrees)')
plt.title('Angular estimation error as a function of target location for ' +modelname)
plt.savefig(dirfiles+'/plot_box_angerror_alltargetlocs_'+modelname+'.png')

# create boxplot for average in six target azimuth bins
angular_error_front = angular_error_pertargetangle[35]+angular_error_pertargetangle[0]+angular_error_pertargetangle[1]
angular_error_Q1 = angular_error_pertargetangle[2]+angular_error_pertargetangle[3]+angular_error_pertargetangle[4]+angular_error_pertargetangle[5]
angular_error_Q2 = angular_error_pertargetangle[6]+angular_error_pertargetangle[7]+angular_error_pertargetangle[8]+angular_error_pertargetangle[9]+angular_error_pertargetangle[10]+angular_error_pertargetangle[11]+angular_error_pertargetangle[12]
angular_error_Q3 = angular_error_pertargetangle[13]+angular_error_pertargetangle[14]+angular_error_pertargetangle[15]+angular_error_pertargetangle[16]
angular_error_back = angular_error_pertargetangle[17]+angular_error_pertargetangle[18]+angular_error_pertargetangle[19]
angular_error_Q4 = angular_error_pertargetangle[20]+angular_error_pertargetangle[21]+angular_error_pertargetangle[22]+angular_error_pertargetangle[23]
angular_error_Q5 = angular_error_pertargetangle[24]+angular_error_pertargetangle[25]+angular_error_pertargetangle[26]+angular_error_pertargetangle[27]+angular_error_pertargetangle[28]+angular_error_pertargetangle[29]+angular_error_pertargetangle[30]
angular_error_Q6 = angular_error_pertargetangle[31]+angular_error_pertargetangle[32]+angular_error_pertargetangle[33]+angular_error_pertargetangle[34]

data = [[angular_error_front],[angular_error_Q1],[angular_error_Q2],[angular_error_Q3],[angular_error_back],[angular_error_Q4],[angular_error_Q5],[angular_error_Q6]]
plt.figure(figsize=(8,15))
flierprops = dict(marker='+', markerfacecolor='black', alpha = .5, markersize=3,
                  linestyle='none')
plt.boxplot(data[5], positions = [1], flierprops=flierprops)
plt.boxplot(data[6], positions = [2], flierprops=flierprops)
plt.boxplot(data[7], positions = [3], flierprops=flierprops)
plt.boxplot(data[0], positions = [4], flierprops=flierprops)
plt.boxplot(data[1], positions = [5], flierprops=flierprops)
plt.boxplot(data[2], positions = [6], flierprops=flierprops)
plt.boxplot(data[3], positions = [7], flierprops=flierprops)
plt.boxplot(data[4], positions = [8], flierprops=flierprops)
ind = np.arange(1,9,1)
plt.yticks(fontsize=20)
plt.xticks(ind,('200-230','240-300','310-340','350-10','20-50','60-120','130-160','170-190'),rotation=90,fontsize=20)
plt.xlabel('Target Locations (degrees)',fontsize=20)
plt.ylabel('Error (degrees)',fontsize=20)
plt.title('Angular estimation error as a function\nof target location for ' +modelname,fontsize=20)
plt.savefig(dirfiles+'/plot_box_angerror_targetlocs_grouped_for_'+modelname+'.png')

# create polarplot of error range
fig = plt.figure()
ax = plt.axes(polar=True)
r = np.ones(len(azimuthrange))
theta = azimuthrange
sizemark = 5*mean_angular_error_pertargetangle
colors = theta
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.scatter(np.radians(theta), r, s = sizemark, c = colors, cmap = 'hsv')
    

# Adavanne's method --> different
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
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Target locations (x) and predicted\nlocations (o) in Cartesian\ncoordinates',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)
plt.savefig(dirfiles+'/plot_scatter_predicted_target_locs_cartcoord_0_90_180_270_for_'+modelname+'.png')

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
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Target locations (x) and predicted \nlocations (o) in Cartesian\n coordinates',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)
plt.savefig(dirfiles+'/plot_scatter_predicted_target_locs_cartcoord_40_140_220_320_for_'+modelname+'.png')


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

plt.figure()
plt.plot(hist.cos_distmet_2D)
plt.plot(hist.val_cos_distmet_2D)
plt.title('Model Cosine Distance')
plt.ylabel('Cosine distance')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
