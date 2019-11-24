# script to compare models

# import libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Specifications
#-----------------------------------------------------------------------------
dirfiles = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\DNN_ESANN'

# specify model names
models = ['model11','model13']

# set azimuth range
azimuthrange = np.arange(0,360,10)

#-----------------------------------------------------------------------------
# Load data
#-----------------------------------------------------------------------------

# load information about validation set
labels_val =  np.load(dirfiles+"/labels_1800_validation.npy")
names_val = pickle.load(open(dirfiles+'/listfilenames_1800_validation.p','rb'))

# load predictions, this will have three dimensions where the first dimension is the model
models_predictions = np.empty([len(models),1800,2])
for x in range(len(models)):
    models_predictions[x] = np.load(dirfiles+"/predictions_"+models[x]+".npy")

#-----------------------------------------------------------------------------
# Calculations
#-----------------------------------------------------------------------------
# retrieve test labels from names 
names_val_angle = np.empty([len(names_val)])
cnt1 = 0
for x in names_val:
    names_val_angle[cnt1] = x[1:4]
    cnt1 += 1
    
# compute mean and standard deviation of error per target azimuth location
mean_prediction = np.empty([len(models),len(azimuthrange),2])
stdev_prediction =  np.empty([len(models),len(azimuthrange),2])
sem_prediction =  np.empty([len(models),len(azimuthrange),2])
for w in range(len(models)):
    for x in range(len(azimuthrange)):
        mean_prediction[w,x] = np.mean(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)
        stdev_prediction[w,x] = np.std(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)
        sem_prediction[w,x] = np.std(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)/np.sqrt(len(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[x])])))

# label per location (complicated way of retrieving it but OK)
mean_label = np.empty([len(azimuthrange),2])
for x in range(len(azimuthrange)):
    mean_label[x] =  np.mean(labels_val[np.where(names_val_angle==azimuthrange[x])],axis=0)

# compute mean, standard deviation, and standard error of the mean per quadrant
front_locs = [-2,-1,0,1,2]
left_locs = [7,8,9,10,11]
right_locs = [25,26,27,28,29]
back_locs = [16,17,18,19,20]
mean_prediction_front = np.empty([len(models),5,2]) # five locations per  quadrant
mean_prediction_left = np.empty([len(models),5,2])
mean_prediction_right = np.empty([len(models),5,2])
mean_prediction_back = np.empty([len(models),5,2])
# first concatenate for each quadrant the locations in a large array and then compute mean and everything else
for x in range(5):
    mean_prediction_front[x] = np.mean(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[[-2,-1,0,1,2]])]),axis=0)

test = azimuthrange[[1,-2]]
#-----------------------------------------------------------------------------
# Figures 
#-----------------------------------------------------------------------------

# create scatterplot of mean prediction per model with error bars for the x and y axis
# to create scatterplots
loc1 = 0
loc2 = 9
loc3 = 18
loc4 = 27
loc5 = 5
loc6 = 13
loc7 = 23
loc8 = 31
color_true = (0,0,0)
color_model1 = (1,0,0)
color_model2 = (0, 0, 1)
fig = plt.figure(figsize=(15,15))
plt.scatter(mean_label[loc1,0],mean_label[loc1,1], s = 200, color=color_true, alpha=1)
plt.scatter(mean_label[loc2,0],mean_label[loc2,1], s = 200, color=color_true, alpha=1)
plt.scatter(mean_label[loc3,0],mean_label[loc3,1], s = 200, color=color_true, alpha=1)
plt.scatter(mean_label[loc4,0],mean_label[loc4,1], s = 200, color=color_true, alpha=1)
plt.scatter(mean_label[loc5,0],mean_label[loc5,1], s = 200, color=color_true, alpha=1)
plt.scatter(mean_label[loc6,0],mean_label[loc6,1], s = 200, color=color_true, alpha=1)
plt.scatter(mean_label[loc7,0],mean_label[loc7,1], s = 200, color=color_true, alpha=1)
plt.scatter(mean_label[loc8,0],mean_label[loc8,1], s = 200, color=color_true, alpha=1)
plt.errorbar(mean_prediction[0,loc1,0],mean_prediction[0,loc1,1], xerr = sem_prediction[0,loc1,0], yerr = sem_prediction[0,loc1,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
plt.errorbar(mean_prediction[1,loc1,0],mean_prediction[1,loc1,1], xerr = sem_prediction[1,loc1,0], yerr = sem_prediction[1,loc1,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc2,0],mean_prediction[0,loc2,1], xerr = sem_prediction[0,loc2,0], yerr = sem_prediction[0,loc2,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
plt.errorbar(mean_prediction[1,loc2,0],mean_prediction[1,loc2,1], xerr = sem_prediction[1,loc2,0], yerr = sem_prediction[1,loc2,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc3,0],mean_prediction[0,loc3,1], xerr = sem_prediction[0,loc3,0], yerr = sem_prediction[0,loc3,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
plt.errorbar(mean_prediction[1,loc3,0],mean_prediction[1,loc3,1], xerr = sem_prediction[1,loc3,0], yerr = sem_prediction[1,loc3,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc4,0],mean_prediction[0,loc4,1], xerr = sem_prediction[0,loc4,0], yerr = sem_prediction[0,loc4,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
plt.errorbar(mean_prediction[1,loc4,0],mean_prediction[1,loc4,1], xerr = sem_prediction[1,loc4,0], yerr = sem_prediction[1,loc4,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc5,0],mean_prediction[0,loc5,1], xerr = sem_prediction[0,loc5,0], yerr = sem_prediction[0,loc5,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
plt.errorbar(mean_prediction[1,loc5,0],mean_prediction[1,loc5,1], xerr = sem_prediction[1,loc5,0], yerr = sem_prediction[1,loc5,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc6,0],mean_prediction[0,loc6,1], xerr = sem_prediction[0,loc6,0], yerr = sem_prediction[0,loc6,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
plt.errorbar(mean_prediction[1,loc6,0],mean_prediction[1,loc6,1], xerr = sem_prediction[1,loc6,0], yerr = sem_prediction[1,loc6,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc7,0],mean_prediction[0,loc7,1], xerr = sem_prediction[0,loc7,0], yerr = sem_prediction[0,loc7,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
plt.errorbar(mean_prediction[1,loc7,0],mean_prediction[1,loc7,1], xerr = sem_prediction[1,loc7,0], yerr = sem_prediction[1,loc7,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc8,0],mean_prediction[0,loc8,1], xerr = sem_prediction[0,loc8,0], yerr = sem_prediction[0,loc8,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
plt.errorbar(mean_prediction[1,loc8,0],mean_prediction[1,loc8,1], xerr = sem_prediction[1,loc8,0], yerr = sem_prediction[1,loc8,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.axis('square')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('Target locations (x) and predicted locations (o) in Cartesian coordinates',fontweight = 'bold', fontsize = 20)
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .5)
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.tick_params(axis='both', which='major', labelsize=15)
#plt.xlabel('x-coordinate',fontsize=15)
#plt.ylabel('y-coordinate',fontsize=15)
plt.savefig(dirfiles+'/plot_scatter_trueandpredicted_acrossmodels.png')    


# create scatterplot of mean prediction per model with error bars for the x and y axis
# to create scatterplots
loc0 = 34
loc1 = 0
loc2 = 2
loc3 = 7
loc4 = 9
loc5 = 11
loc6 = 16
loc7 = 18
loc8 = 20
loc9 = 25
loc10 = 27
loc11 = 29
color_true1 = (0,0,0,1) # the last axis is the face alpha
color_true2 = (0,0,0,.4)
color_true3 = (0,0,0,0)
color_model1_1 = (1,0,0,1)
color_model1_2 = (1,0,0,.4)
color_model1_3 = (1,0,0,0)
color_model2_1 = (0, 0, 1,1)
color_model2_2 = (0, 0, 1,.4)
color_model2_3 = (0, 0, 1,0)

fig = plt.figure(figsize=(15,15))
plt.scatter(mean_label[loc0,0],mean_label[loc0,1], s = 200, marker = 'o', edgecolors = color_true1, linewidths = 2, color = color_true1)
plt.scatter(mean_label[loc1,0],mean_label[loc1,1], s = 200, marker = 'o', edgecolors = color_true1, linewidths = 2, color = color_true2)
plt.scatter(mean_label[loc2,0],mean_label[loc2,1], s = 200, marker = 'o', edgecolors = color_true1, linewidths = 2, color = color_true3)
plt.scatter(mean_label[loc3,0],mean_label[loc3,1], s = 200, marker = '^', edgecolors = color_true1, linewidths = 2, color = color_true1)
plt.scatter(mean_label[loc4,0],mean_label[loc4,1], s = 200, marker = '^', edgecolors = color_true1, linewidths = 2, color = color_true2)
plt.scatter(mean_label[loc5,0],mean_label[loc5,1], s = 200, marker = '^', edgecolors = color_true1, linewidths = 2, color = color_true3)
plt.scatter(mean_label[loc6,0],mean_label[loc6,1], s = 200, marker = 's', edgecolors = color_true1, linewidths = 2, color = color_true1)
plt.scatter(mean_label[loc7,0],mean_label[loc7,1], s = 200, marker = 's', edgecolors = color_true1, linewidths = 2, color = color_true2)
plt.scatter(mean_label[loc8,0],mean_label[loc8,1], s = 200, marker = 's', edgecolors = color_true1, linewidths = 2, color = color_true3)
plt.scatter(mean_label[loc9,0],mean_label[loc9,1], s = 200, marker = 'D', edgecolors = color_true1, linewidths = 2, color = color_true1)
plt.scatter(mean_label[loc10,0],mean_label[loc10,1], s = 200, marker = 'D', edgecolors = color_true1, linewidths = 2, color = color_true2)
plt.scatter(mean_label[loc11,0],mean_label[loc11,1], s = 200, marker = 'D', edgecolors = color_true1, linewidths = 2, color = color_true3)
#model 1
plt.errorbar(mean_prediction[0,loc0,0],mean_prediction[0,loc0,1], xerr = sem_prediction[0,loc0,0], yerr = sem_prediction[0,loc0,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_1)
plt.errorbar(mean_prediction[0,loc1,0],mean_prediction[0,loc1,1], xerr = sem_prediction[0,loc1,0], yerr = sem_prediction[0,loc1,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_2)
plt.errorbar(mean_prediction[0,loc2,0],mean_prediction[0,loc2,1], xerr = sem_prediction[0,loc2,0], yerr = sem_prediction[0,loc2,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_3)

plt.errorbar(mean_prediction[0,loc3,0],mean_prediction[0,loc3,1], xerr = sem_prediction[0,loc3,0], yerr = sem_prediction[0,loc3,1], fmt = '^', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_1)
plt.errorbar(mean_prediction[0,loc4,0],mean_prediction[0,loc4,1], xerr = sem_prediction[0,loc4,0], yerr = sem_prediction[0,loc4,1], fmt = '^', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_2)
plt.errorbar(mean_prediction[0,loc5,0],mean_prediction[0,loc5,1], xerr = sem_prediction[0,loc5,0], yerr = sem_prediction[0,loc5,1], fmt = '^', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_3)

plt.errorbar(mean_prediction[0,loc6,0],mean_prediction[0,loc6,1], xerr = sem_prediction[0,loc6,0], yerr = sem_prediction[0,loc6,1], fmt = 's', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_1)
plt.errorbar(mean_prediction[0,loc7,0],mean_prediction[0,loc7,1], xerr = sem_prediction[0,loc7,0], yerr = sem_prediction[0,loc7,1], fmt = 's', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_2)
plt.errorbar(mean_prediction[0,loc8,0],mean_prediction[0,loc8,1], xerr = sem_prediction[0,loc8,0], yerr = sem_prediction[0,loc8,1], fmt = 's', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_3)

plt.errorbar(mean_prediction[0,loc9,0],mean_prediction[0,loc9,1], xerr = sem_prediction[0,loc9,0], yerr = sem_prediction[0,loc9,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_1)
plt.errorbar(mean_prediction[0,loc10,0],mean_prediction[0,loc10,1], xerr = sem_prediction[0,loc10,0], yerr = sem_prediction[0,loc10,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_2)
plt.errorbar(mean_prediction[0,loc11,0],mean_prediction[0,loc11,1], xerr = sem_prediction[0,loc11,0], yerr = sem_prediction[0,loc11,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model1_1, ecolor = color_model1_1, linewidth = 2, color=color_model1_3)
#model2
plt.errorbar(mean_prediction[1,loc0,0],mean_prediction[1,loc0,1], xerr = sem_prediction[1,loc0,0], yerr = sem_prediction[1,loc0,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_1)
plt.errorbar(mean_prediction[1,loc1,0],mean_prediction[1,loc1,1], xerr = sem_prediction[1,loc1,0], yerr = sem_prediction[1,loc1,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_2)
plt.errorbar(mean_prediction[1,loc2,0],mean_prediction[1,loc2,1], xerr = sem_prediction[1,loc2,0], yerr = sem_prediction[1,loc2,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_3)

plt.errorbar(mean_prediction[1,loc3,0],mean_prediction[1,loc3,1], xerr = sem_prediction[1,loc3,0], yerr = sem_prediction[1,loc3,1], fmt = '^', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_1)
plt.errorbar(mean_prediction[1,loc4,0],mean_prediction[1,loc4,1], xerr = sem_prediction[1,loc4,0], yerr = sem_prediction[1,loc4,1], fmt = '^', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_2)
plt.errorbar(mean_prediction[1,loc5,0],mean_prediction[1,loc5,1], xerr = sem_prediction[1,loc5,0], yerr = sem_prediction[1,loc5,1], fmt = '^', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_3)

plt.errorbar(mean_prediction[1,loc6,0],mean_prediction[1,loc6,1], xerr = sem_prediction[1,loc6,0], yerr = sem_prediction[1,loc6,1], fmt = 's', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_1)
plt.errorbar(mean_prediction[1,loc7,0],mean_prediction[1,loc7,1], xerr = sem_prediction[1,loc7,0], yerr = sem_prediction[1,loc7,1], fmt = 's', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_2)
plt.errorbar(mean_prediction[1,loc8,0],mean_prediction[1,loc8,1], xerr = sem_prediction[1,loc8,0], yerr = sem_prediction[1,loc8,1], fmt = 's', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_3)

plt.errorbar(mean_prediction[1,loc9,0],mean_prediction[1,loc9,1], xerr = sem_prediction[1,loc9,0], yerr = sem_prediction[1,loc9,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_1)
plt.errorbar(mean_prediction[1,loc10,0],mean_prediction[1,loc10,1], xerr = sem_prediction[1,loc10,0], yerr = sem_prediction[1,loc10,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_2)
plt.errorbar(mean_prediction[1,loc11,0],mean_prediction[1,loc11,1], xerr = sem_prediction[1,loc11,0], yerr = sem_prediction[1,loc11,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model2_1, ecolor = color_model2_1, linewidth = 2, color=color_model2_3)

plt.axis('square')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.xticks(ticks = None)
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .5)
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.tick_params(axis='both', which='major', labelsize=0)
#plt.xlabel('x-coordinate',fontsize=15)
#plt.ylabel('y-coordinate',fontsize=15)
plt.savefig(dirfiles+'/plot_scatter_trueandpredicted_perquadrant_acrossmodels.png')    
