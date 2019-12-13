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
from CustLoss_cosine_distance_angular import cos_dist_2D_angular
from CustMet_cosine_distance_angular import cos_distmet_2D_angular


# define name of current model
modelname = "model19"

# model parameters for evaluation
sizebatches = 128

# set azimuth range
azimuthrange = np.arange(0,360,10)

#------------------------------------------------------------------------------
# Preparations
#------------------------------------------------------------------------------

# load model
model = load_model(dirfiles+'/'+modelname+'_final.h5', custom_objects={"cust_mean_squared_error": cust_mean_squared_error, "cos_dist_2D_angular": cos_dist_2D_angular, "cos_distmet_2D_angular": cos_distmet_2D_angular})

# load history of the model
hist = pandas.read_csv(dirfiles+"/history_"+modelname+".csv")

# load the data
an_l_val = np.load(dirfiles+"/an_l_1800_validation.npy")
an_r_val = np.load(dirfiles+"/an_r_1800_validation.npy") 
labels_val =  np.load(dirfiles+"/labels_1800_validation.npy")
names_val = pickle.load(open(dirfiles+'/listfilenames_1800_validation.p','rb'))

# add a fourth dimension to the data
an_l_val = np.expand_dims(an_l_val,axis = 3)
an_r_val = np.expand_dims(an_r_val,axis = 3)

# prepare data for model evaluation
X_test = [an_l_val, an_r_val]
Y_test = labels_val

#------------------------------------------------------------------------------
# Run model evaluation
#------------------------------------------------------------------------------

# predict --> create predictions (set to 1) or load predictions (set to 0)
newpredictions = 0
if newpredictions == 1:
    predictions = generate_model_predictions(model, X_test, modelname, dirfiles, sizebatches)
elif newpredictions == 0:
    predictions = np.load(dirfiles+"/predictions_"+modelname+".npy")
        
# evaluate the model on unseen data (shouldn't be test data)
score = model.evaluate(X_test, Y_test, verbose=1) # !!!! note that the cosine distance computation is wrong and cannot be used,
# should use the cosine_distance_degrees average instead

# this was used to check whether the model predictions match the model evaluation. 
#cos_sim = np.sum(Y_test*predictions, axis=1)/(np.sqrt(np.sum(np.square(Y_test),axis=1))*np.sqrt(np.sum(np.square(predictions),axis=1)))

#------------------------------------------------------------------------------
# Analyze predictions
#------------------------------------------------------------------------------

# retrieve test labels from names 
names_val_angle = np.empty([len(names_val)])
cnt1 = 0
for x in names_val:
    names_val_angle[cnt1] = x[1:4]
    cnt1 += 1

  
# compute angular error  
cosine_distance_degrees = np.empty([len(predictions)])
for  x in range(len(predictions)):
     cossim = np.sum(labels_val[x]*predictions[x])/(np.sqrt(np.sum(np.square(labels_val[x])))*np.sqrt(np.sum(np.square(predictions[x]))))
     cosine_distance_degrees[x] = np.arccos(cossim)*180/np.pi
     
# compute mean and standard deviation of error per target azimuth location
mean_angular_error_pertargetangle = np.empty([len(azimuthrange)]) 
stdev_angular_error_pertagetanle = np.empty([len(azimuthrange)]) 
angular_error_pertargetangle = [None] *len(azimuthrange)
mean_prediction = np.empty([len(azimuthrange),2])
stdev_prediction =  np.empty([len(azimuthrange),2])
mean_label = np.empty([len(azimuthrange),2])
for x in range(len(azimuthrange)):
    mean_angular_error_pertargetangle[x] = np.mean(cosine_distance_degrees[np.where(names_val_angle==azimuthrange[x])])
    stdev_angular_error_pertagetanle[x] = np.std(cosine_distance_degrees[names_val_angle==azimuthrange[x]])
    angular_error_pertargetangle[x] = np.ndarray.tolist(cosine_distance_degrees[names_val_angle==azimuthrange[x]])
    mean_prediction[x] = np.mean(predictions[np.where(names_val_angle==azimuthrange[x])],axis=0)
    stdev_prediction[x] = np.std(predictions[np.where(names_val_angle==azimuthrange[x])],axis=0)
    mean_label[x] =  np.mean(labels_val[np.where(names_val_angle==azimuthrange[x])],axis=0)
    
#------------------------------------------------------------------------------
# Create figures
#------------------------------------------------------------------------------    

# create boxplot for all targetazimuths
# change order of azms such that 0 deg is in the middle
data = angular_error_pertargetangle[-18:]+angular_error_pertargetangle[:18]
# remove nans from 'data' 
for nancheck in range(len(data)):
    tempdata = data[nancheck]
    tempdata =  [x for x in tempdata if str(x) != 'nan']
    data[nancheck] = tempdata
plt.figure()
flierprops = dict(marker='+', markerfacecolor='black', alpha = .5, markersize=3, linestyle='none')
plt.boxplot(data, flierprops=flierprops)
ind = np.arange(1,37,2)
plt.xticks(ind,('180','200','220','240','260','280','300','320','340','0','20','40','60','80','100','120','140','160'),rotation=90)

#plt.xticks(ind,('180','190','200','210','220','230','240','250','260','270','280','290','300','310','320','330','340','350','0','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150','160','170'))
plt.xlabel('Target Locations (degrees)')
plt.ylabel('Error (degrees)')
plt.title('Angular estimation error as a function of target location for ' +modelname)
plt.savefig(dirfiles+'/plot_box_angerror_alltargetlocs_'+modelname+'.png')

# create scatterplot of real and mean prediction 
plt.figure()
plt.scatter(mean_prediction[:,0],mean_prediction[:,1],c=azimuthrange, alpha=1, cmap = 'hsv')
plt.scatter(mean_label[:,0],mean_label[:,1],c=azimuthrange, alpha=1, cmap = 'hsv')
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Target locations (x) and predicted\nlocations (o) in Cartesian\ncoordinates',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)


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
plt.scatter(predictions[np.squeeze(names_val_angle==anglecheck1),0],predictions[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(np.mean(predictions[np.squeeze(names_val_angle==anglecheck1),0]),np.mean(predictions[np.squeeze(names_val_angle==anglecheck1),1]),color=color1, alpha=1, marker = "o",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_val_angle==anglecheck2),0],predictions[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(np.mean(predictions[np.squeeze(names_val_angle==anglecheck2),0]),np.mean(predictions[np.squeeze(names_val_angle==anglecheck2),1]),color=color2, alpha=1, marker = "o",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_val_angle==anglecheck3),0],predictions[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(np.mean(predictions[np.squeeze(names_val_angle==anglecheck3),0]),np.mean(predictions[np.squeeze(names_val_angle==anglecheck3),1]),color=color3, alpha=1, marker = "o",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_val_angle==anglecheck4),0],predictions[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(np.mean(predictions[np.squeeze(names_val_angle==anglecheck4),0]),np.mean(predictions[np.squeeze(names_val_angle==anglecheck4),1]),color=color4, alpha=1, marker = "o",s=100, edgecolors="k",linewidth=1)
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
plt.scatter(predictions[np.squeeze(names_val_angle==anglecheck1),0],predictions[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(np.mean(predictions[np.squeeze(names_val_angle==anglecheck1),0]),np.mean(predictions[np.squeeze(names_val_angle==anglecheck1),1]),color=color1, alpha=1, marker = "o",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_val_angle==anglecheck2),0],predictions[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(np.mean(predictions[np.squeeze(names_val_angle==anglecheck2),0]),np.mean(predictions[np.squeeze(names_val_angle==anglecheck2),1]),color=color2, alpha=1, marker = "o",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_val_angle==anglecheck3),0],predictions[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(np.mean(predictions[np.squeeze(names_val_angle==anglecheck3),0]),np.mean(predictions[np.squeeze(names_val_angle==anglecheck3),1]),color=color3, alpha=1, marker = "o",s=100, edgecolors="k",linewidth=1)
plt.scatter(predictions[np.squeeze(names_val_angle==anglecheck4),0],predictions[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(np.mean(predictions[np.squeeze(names_val_angle==anglecheck4),0]),np.mean(predictions[np.squeeze(names_val_angle==anglecheck4),1]),color=color4, alpha=1, marker = "o",s=100, edgecolors="k",linewidth=1)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Target locations (x) and predicted \nlocations (o) in Cartesian\n coordinates',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)
plt.savefig(dirfiles+'/plot_scatter_predicted_target_locs_cartcoord_40_140_220_320_for_'+modelname+'.png')



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
plt.title("Model 19: Training and validation loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train loss", "val loss"], loc="upper right")
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
