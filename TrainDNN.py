# Main to train a DNN model

dir_fufiles = "/home/jovyan/DNN_ESANN_dev" # specify directory where function files are located
dir_mofiles = "/home/jovyan/DNN_ESANN_dev" # specify directory where model files are located 

#dir_mofiles = r"C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning"

# import packages, libraries, functions to call them later
import ImportAndPrepare_Data
from pytictoc import TicToc
t = TicToc()
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform
from CustLoss_MSE import cust_mean_squared_error
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint

csv_loss_logger = CSVLogger('history_model5.csv')
dir_model_logger = ModelCheckpoint(dir_mofiles+"/model5.h5")
model_logger = ModelCheckpoint(dir_model_logger,  monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)


# load data
labels_rand_train, labels_rand_test, an_l_rand_train, an_l_rand_test, an_r_rand_train, an_r_rand_test = ImportAndPrepare_Data.im_and_prep()

# load model
t.tic()
mymodel = load_model(dir_mofiles+"/DNN_model5.h5",custom_objects={'GlorotUniform': glorot_uniform(), "cust_mean_squared_error": cust_mean_squared_error})
mymodel.summary()
t.toc("loading the model took ")



# train the model
t.tic()
history = mymodel.fit([an_l_rand_train, an_r_rand_train], labels_rand_train, validation_data=((an_l_rand_test,an_r_rand_test),labels_rand_test), epochs = 20, batch_size = 64, verbose = 1, use_multiprocessing = True, callbacks = [csv_loss_logger])
t.toc("training the model took ")

mymodel.save("model5_final.h5")


print("Save model")




# =============================================================================
# # metrics to save from the model
# hist_csv_file = 'history_model3_soundssmall.csv'
# with open(hist_csv_file, mode='w') as f:
#     history.to_csv(f)
# =============================================================================

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
