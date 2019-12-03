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
#models = ['model11','model13','model15', 'model16']
models = ['model16','model19']

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

# retrieve whether it is anechoic or reverb, if reverb code 1
names_val_env = np.zeros([len(names_val)])
cnt2 = 0
for x in names_val:
    if x.find('RI02') != -1:
        names_val_env[cnt2] = 1
    cnt2 += 1


# compute cosine distance in degrees
dims_matrix_cosdist = np.shape(models_predictions)
cosine_distance_degrees = np.empty([dims_matrix_cosdist[0],dims_matrix_cosdist[1]])
for w in range(dims_matrix_cosdist[0]):
    for  x in range(dims_matrix_cosdist[1]):
         cossim = np.sum(labels_val[x]*models_predictions[w,x])/(np.sqrt(np.sum(np.square(labels_val[x])))*np.sqrt(np.sum(np.square(models_predictions[w,x]))))
         cosine_distance_degrees[w,x] = np.arccos(cossim)*180/np.pi
         
# now you want to compute the cosine distance as a function of azimuth location
mean_cosdistdeg_az = np.empty([dims_matrix_cosdist[0],len(azimuthrange)])
stdev_cosdistdeg_az = np.empty([dims_matrix_cosdist[0],len(azimuthrange)])
sem_cosdistdeg_az = np.empty([dims_matrix_cosdist[0],len(azimuthrange)])
cosdistdeg_az = np.empty([dims_matrix_cosdist[0],len(azimuthrange),50]) # hardcoded 50because the number of sounds in the eval set is 50
for w in range(dims_matrix_cosdist[0]):
    for x in range(len(azimuthrange)):
        mean_cosdistdeg_az[w,x] = np.mean(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x])]))
        stdev_cosdistdeg_az[w,x] = np.std(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x])]))
        sem_cosdistdeg_az[w,x] = np.std(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x])]))/np.sqrt(len(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x])])))
        cosdistdeg_az[w,x,] = np.ndarray.tolist(cosine_distance_degrees[w,names_val_angle==azimuthrange[x]])

# now you want to compute the cosine distance as a function of env
env_range = [0,1]
mean_cosdistdeg_env = np.empty([dims_matrix_cosdist[0],len(env_range)]) # 2 for the number of environments that you have
stdev_cosdistdeg_env = np.empty([dims_matrix_cosdist[0],len(env_range)])
for w in range(dims_matrix_cosdist[0]):
    for x in range(len(env_range)):
        mean_cosdistdeg_env[w,x] = np.mean(np.squeeze(cosine_distance_degrees[w,np.where(names_val_env==env_range[x])]))
        stdev_cosdistdeg_env[w,x] = np.std(np.squeeze(cosine_distance_degrees[w,np.where(names_val_env==env_range[x])]))

# =============================================================================
# # now you want to compute the cosine distance as a function of azimuth and env
# mean_cosdistdeg_az_env = np.empty([dims_matrix_cosdist[0],len(azimuthrange)])
# stdev_cosdistdeg_az_env = np.empty([dims_matrix_cosdist[0],len(azimuthrange)])
# for w in range(dims_matrix_cosdist[0]):
#     for wx in range(len(env_range)):
#         for x in range(len(azimuthrange)):
#             mean_cosdistdeg_az_env[w,wx,x] = np.mean(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x] and names_val_env==env_range[wx])]))
#             stdev_cosdistdeg_az_env[w,wx,x] = np.std(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x]) and np.where(names_val_env==env_range[wx])]))
# =============================================================================
  
# compute mean and standard deviation of error per target azimuth location
mean_prediction = np.empty([len(models),len(azimuthrange),2])
stdev_prediction =  np.empty([len(models),len(azimuthrange),2])
sem_prediction =  np.empty([len(models),len(azimuthrange),2])
for w in range(len(models)):
    for x in range(len(azimuthrange)):
        mean_prediction[w,x] = np.mean(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)
        stdev_prediction[w,x] = np.std(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)
        sem_prediction[w,x] = np.std(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)/np.sqrt(len(np.squeeze(models_predictions[w,np.where(names_val_angle==azimuthrange[x])])))
     
# convert mean prediction to mean predicted angle
mean_predangles = np.empty([len(models),len(azimuthrange)])
for w in range(len(models)):
    for x in range(len(azimuthrange)):
        tempx = mean_prediction[w,x,0]
        tempy = mean_prediction[w,x,1]
        if tempx >=0 and tempy >= 0:
            preddeg = np.arcsin(mean_prediction[w,x,0]/np.sqrt(np.square(mean_prediction[w,x,0])+np.square(mean_prediction[w,x,1])))*180/np.pi
        elif tempx >=0 and tempy < 0:
            preddeg = 180-np.arcsin(mean_prediction[w,x,0]/np.sqrt(np.square(mean_prediction[w,x,0])+np.square(mean_prediction[w,x,1])))*180/np.pi
        elif tempx< 0 and tempy < 0:
            preddeg = 180+np.abs(np.arcsin(mean_prediction[w,x,0]/np.sqrt(np.square(mean_prediction[w,x,0])+np.square(mean_prediction[w,x,1]))))*180/np.pi
        elif tempx< 0 and tempy >=0:
            preddeg = 360+np.arcsin(mean_prediction[w,x,0]/np.sqrt(np.square(mean_prediction[w,x,0])+np.square(mean_prediction[w,x,1])))*180/np.pi
        mean_predangles[w,x] = preddeg

# label per location (complicated way of retrieving it but OK)
mean_label = np.empty([len(azimuthrange),2])
for x in range(len(azimuthrange)):
    mean_label[x] =  np.mean(labels_val[np.where(names_val_angle==azimuthrange[x])],axis=0)


#create logicals (booleans of the locations that should be grouped together)
# frontal: 320 330 340 350 0 10 20 30 40
# right: 50 60 70 80 90 100 110 120 130 
# behind: 140 150 160 170 180 190 200 210 220 
# left: 230 240 250 260 270 280 290 300 310     
q_front = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
q_right = [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
q_back = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
q_left = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0]
q_front_narrow = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]
q_right_narrow = [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
q_back_narrow = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
q_left_narrow = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0]

# turn into boolean
q_front = [bool(x) for x in q_front]
q_right = [bool(x) for x in q_right]
q_back = [bool(x) for x in q_back]
q_left = [bool(x) for x in q_left]
q_front_narrow = [bool(x) for x in q_front_narrow]

# compute the means
q_front_mean_cosdistdeg = np.mean(mean_cosdistdeg_az[0,q_front])
q_right_mean_cosdistdeg = np.mean(mean_cosdistdeg_az[0,q_right])
q_back_mean_cosdistdeg = np.mean(mean_cosdistdeg_az[0,q_back])
q_left_mean_cosdistdeg = np.mean(mean_cosdistdeg_az[0,q_left])
q_front_narrow_mean_cosdistdeg = np.mean(mean_cosdistdeg_az[0,q_front_narrow])


#-----------------------------------------------------------------------------
# Figures 
#-----------------------------------------------------------------------------

### plot model predictions on polar plot
# model 1
theta_pred = np.squeeze(mean_predangles[0,])
r_pred = np.ones(len(np.squeeze(mean_predangles[0,])))
theta_true = azimuthrange
r_true = np.ones(len(azimuthrange))*1.5   
colors_pred = azimuthrange
colors_true = azimuthrange
fig = plt.figure()
ax = fig.add_subplot(111,projection = 'polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rorigin(-2.5)
ax.set_rlim(.5,2)
ax.set_yticklabels([])
ax.grid(linewidth = .75, linestyle = ':')
c = ax.scatter(np.radians(theta_pred), r_pred, c = colors_pred, cmap = 'jet')
c = ax.scatter(np.radians(theta_true), r_true, marker = '^', c = colors_true, cmap = 'jet', alpha = 1)
plt.savefig(dirfiles+'/plot_ESANN_polar_trueandpredicted_modelMSE_m16.eps')    

# model 2
theta_pred = np.squeeze(mean_predangles[1,])
r_pred = np.ones(len(np.squeeze(mean_predangles[0,])))
theta_true = azimuthrange
r_true = np.ones(len(azimuthrange))*1.5   
colors_pred = azimuthrange
colors_true = azimuthrange
fig = plt.figure()
ax = fig.add_subplot(111,projection = 'polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rorigin(-2.5)
ax.set_rlim(.5,2)
ax.set_yticklabels([])
ax.grid(linewidth = .75, linestyle = ':')
c = ax.scatter(np.radians(theta_pred), r_pred, c = colors_pred, cmap = 'jet')
c = ax.scatter(np.radians(theta_true), r_true, marker = '^', c = colors_true, cmap = 'jet', alpha = 1)
plt.savefig(dirfiles+'/plot_ESANN_polar_trueandpredicted_modelAD_m19.eps')    

### create a boxplot of the mean error per environment
flierspecs = dict(marker = '+', markersize = 2)
data_model1 = [np.squeeze(cosine_distance_degrees[0,np.where(names_val_env==0)]),np.squeeze(cosine_distance_degrees[0,np.where(names_val_env==1)])]
data_model2 = [np.squeeze(cosine_distance_degrees[1,np.where(names_val_env==0)]),np.squeeze(cosine_distance_degrees[1,np.where(names_val_env==1)])]
plt.figure(figsize=[10,6])
box1 = plt.boxplot(data_model1, positions=[1,1.5], flierprops = flierspecs, notch=True, patch_artist=True)
box2 = plt.boxplot(data_model2, positions=[2,2.5], flierprops = flierspecs,notch=True, patch_artist=True)
# change face color of model 1
for box in box1['boxes']:
    box.set(facecolor='green')
plt.ylabel('Angular distance (degrees)', fontsize = 12)
plt.yticks(fontsize = 12)
plt.xticks([1, 1.5, 2, 2.5],['Anechoic','Hall','Anechoic','Hall'],fontsize = 12)
plt.legend([box1["boxes"][0], box2["boxes"][0]], ['MSE', 'AD'], loc='upper right', fontsize = 12)
plt.savefig(dirfiles+'/plot_ESANN_box_predictionerror_perenvironment_m16_m19.eps')


### create boxplot of mean error per azimuth position
plt.figure(figsize=[10,6])
data_model1_az = np.squeeze(mean_cosdistdeg_az[0,])
sem_model1_az = sem_cosdistdeg_az[0,]
data_model2_az = np.squeeze(mean_cosdistdeg_az[1,])
sem_model2_az = sem_cosdistdeg_az[1,]

idx = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
plt.errorbar(np.linspace(1,36,36),data_model1_az[idx],yerr=sem_model1_az[idx],fmt= 'o', alpha = .5)
plt.errorbar(np.linspace(1,36,36),data_model2_az[idx],yerr=sem_model2_az[idx],fmt= 'o', alpha = .5)
plt.xticks(np.linspace(1,36,18),('180','200','220','240','260','280','300','320','340','0','20','40','60','80','100','120','140','160'),rotation=60,fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(['MSE', 'AD'], loc='upper right', fontsize = 12)
plt.xlabel('Target azimuth position (degrees)', fontsize = 12)
plt.ylabel('Angular distance (degrees)', fontsize = 12)
plt.grid(linestyle = ':', alpha = .7)
plt.savefig(dirfiles+'/plot_ESANN_angulardistance_perazimuth_bothmodels_m16_m19.eps')
#-----------------------------------------------------------------------------
# OLD CODE
#-----------------------------------------------------------------------------

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
# =============================================================================
# color_model2 = (0, 0, 1)
# color_model3 = (0,1,0)
# =============================================================================
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
#plt.errorbar(mean_prediction[1,loc1,0],mean_prediction[1,loc1,1], xerr = sem_prediction[1,loc1,0], yerr = sem_prediction[1,loc1,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc2,0],mean_prediction[0,loc2,1], xerr = sem_prediction[0,loc2,0], yerr = sem_prediction[0,loc2,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
#plt.errorbar(mean_prediction[1,loc2,0],mean_prediction[1,loc2,1], xerr = sem_prediction[1,loc2,0], yerr = sem_prediction[1,loc2,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc3,0],mean_prediction[0,loc3,1], xerr = sem_prediction[0,loc3,0], yerr = sem_prediction[0,loc3,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
#plt.errorbar(mean_prediction[1,loc3,0],mean_prediction[1,loc3,1], xerr = sem_prediction[1,loc3,0], yerr = sem_prediction[1,loc3,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc4,0],mean_prediction[0,loc4,1], xerr = sem_prediction[0,loc4,0], yerr = sem_prediction[0,loc4,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
#plt.errorbar(mean_prediction[1,loc4,0],mean_prediction[1,loc4,1], xerr = sem_prediction[1,loc4,0], yerr = sem_prediction[1,loc4,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc5,0],mean_prediction[0,loc5,1], xerr = sem_prediction[0,loc5,0], yerr = sem_prediction[0,loc5,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
#plt.errorbar(mean_prediction[1,loc5,0],mean_prediction[1,loc5,1], xerr = sem_prediction[1,loc5,0], yerr = sem_prediction[1,loc5,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc6,0],mean_prediction[0,loc6,1], xerr = sem_prediction[0,loc6,0], yerr = sem_prediction[0,loc6,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
#plt.errorbar(mean_prediction[1,loc6,0],mean_prediction[1,loc6,1], xerr = sem_prediction[1,loc6,0], yerr = sem_prediction[1,loc6,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc7,0],mean_prediction[0,loc7,1], xerr = sem_prediction[0,loc7,0], yerr = sem_prediction[0,loc7,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
#plt.errorbar(mean_prediction[1,loc7,0],mean_prediction[1,loc7,1], xerr = sem_prediction[1,loc7,0], yerr = sem_prediction[1,loc7,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
plt.errorbar(mean_prediction[0,loc8,0],mean_prediction[0,loc8,1], xerr = sem_prediction[0,loc8,0], yerr = sem_prediction[0,loc8,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model1, alpha=0.4)
#plt.errorbar(mean_prediction[1,loc8,0],mean_prediction[1,loc8,1], xerr = sem_prediction[1,loc8,0], yerr = sem_prediction[1,loc8,1], fmt = 'o', markersize = 10, linewidth = 4, color=color_model2, alpha=0.4)
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
# =============================================================================
# color_model3_1 = (0, 128/255, 15/255,1)
# color_model3_2 = (0, 128/255, 15/255,.4)
# color_model3_3 = (0, 128/255, 15/255,0)
# color_model4_1 = (1, 162/255, 0,1)
# color_model4_2 = (1, 162/255, 0,.4)
# color_model4_3 = (1, 162/255, 0,0)
# 
# =============================================================================
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

# =============================================================================
# #model3
# plt.errorbar(mean_prediction[2,loc0,0],mean_prediction[2,loc0,1], xerr = sem_prediction[2,loc0,0], yerr = sem_prediction[2,loc0,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_1)
# plt.errorbar(mean_prediction[2,loc1,0],mean_prediction[2,loc1,1], xerr = sem_prediction[2,loc1,0], yerr = sem_prediction[2,loc1,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_2)
# plt.errorbar(mean_prediction[2,loc2,0],mean_prediction[2,loc2,1], xerr = sem_prediction[2,loc2,0], yerr = sem_prediction[2,loc2,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_3)
# 
# plt.errorbar(mean_prediction[2,loc3,0],mean_prediction[2,loc3,1], xerr = sem_prediction[2,loc3,0], yerr = sem_prediction[2,loc3,1], fmt = '^', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_1)
# plt.errorbar(mean_prediction[2,loc4,0],mean_prediction[2,loc4,1], xerr = sem_prediction[2,loc4,0], yerr = sem_prediction[2,loc4,1], fmt = '^', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_2)
# plt.errorbar(mean_prediction[2,loc5,0],mean_prediction[2,loc5,1], xerr = sem_prediction[2,loc5,0], yerr = sem_prediction[2,loc5,1], fmt = '^', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_3)
# 
# plt.errorbar(mean_prediction[2,loc6,0],mean_prediction[2,loc6,1], xerr = sem_prediction[2,loc6,0], yerr = sem_prediction[2,loc6,1], fmt = 's', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_1)
# plt.errorbar(mean_prediction[2,loc7,0],mean_prediction[2,loc7,1], xerr = sem_prediction[2,loc7,0], yerr = sem_prediction[2,loc7,1], fmt = 's', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_2)
# plt.errorbar(mean_prediction[2,loc8,0],mean_prediction[2,loc8,1], xerr = sem_prediction[2,loc8,0], yerr = sem_prediction[2,loc8,1], fmt = 's', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_3)
# 
# plt.errorbar(mean_prediction[2,loc9,0],mean_prediction[2,loc9,1], xerr = sem_prediction[2,loc9,0], yerr = sem_prediction[2,loc9,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_1)
# plt.errorbar(mean_prediction[2,loc10,0],mean_prediction[2,loc10,1], xerr = sem_prediction[2,loc10,0], yerr = sem_prediction[2,loc10,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_2)
# plt.errorbar(mean_prediction[2,loc11,0],mean_prediction[2,loc11,1], xerr = sem_prediction[2,loc11,0], yerr = sem_prediction[2,loc11,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model3_1, ecolor = color_model3_1, linewidth = 2, color=color_model3_3)
# 
# #model4
# plt.errorbar(mean_prediction[3,loc0,0],mean_prediction[3,loc0,1], xerr = sem_prediction[3,loc0,0], yerr = sem_prediction[3,loc0,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_1)
# plt.errorbar(mean_prediction[3,loc1,0],mean_prediction[3,loc1,1], xerr = sem_prediction[3,loc1,0], yerr = sem_prediction[3,loc1,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_2)
# plt.errorbar(mean_prediction[3,loc2,0],mean_prediction[3,loc2,1], xerr = sem_prediction[3,loc2,0], yerr = sem_prediction[3,loc2,1], fmt = 'o', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_3)
# 
# plt.errorbar(mean_prediction[3,loc3,0],mean_prediction[3,loc3,1], xerr = sem_prediction[3,loc3,0], yerr = sem_prediction[3,loc3,1], fmt = '^', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_1)
# plt.errorbar(mean_prediction[3,loc4,0],mean_prediction[3,loc4,1], xerr = sem_prediction[3,loc4,0], yerr = sem_prediction[3,loc4,1], fmt = '^', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_2)
# plt.errorbar(mean_prediction[3,loc5,0],mean_prediction[3,loc5,1], xerr = sem_prediction[3,loc5,0], yerr = sem_prediction[3,loc5,1], fmt = '^', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_3)
# 
# plt.errorbar(mean_prediction[3,loc6,0],mean_prediction[3,loc6,1], xerr = sem_prediction[3,loc6,0], yerr = sem_prediction[3,loc6,1], fmt = 's', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_1)
# plt.errorbar(mean_prediction[3,loc7,0],mean_prediction[3,loc7,1], xerr = sem_prediction[3,loc7,0], yerr = sem_prediction[3,loc7,1], fmt = 's', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_2)
# plt.errorbar(mean_prediction[3,loc8,0],mean_prediction[3,loc8,1], xerr = sem_prediction[3,loc8,0], yerr = sem_prediction[3,loc8,1], fmt = 's', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_3)
# 
# plt.errorbar(mean_prediction[3,loc9,0],mean_prediction[3,loc9,1], xerr = sem_prediction[3,loc9,0], yerr = sem_prediction[3,loc9,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_1)
# plt.errorbar(mean_prediction[3,loc10,0],mean_prediction[3,loc10,1], xerr = sem_prediction[3,loc10,0], yerr = sem_prediction[3,loc10,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_2)
# plt.errorbar(mean_prediction[3,loc11,0],mean_prediction[3,loc11,1], xerr = sem_prediction[3,loc11,0], yerr = sem_prediction[3,loc11,1], fmt = 'D', markersize = 15, markeredgecolor =  color_model4_1, ecolor = color_model4_1, linewidth = 2, color=color_model4_3)
# 
# =============================================================================

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
