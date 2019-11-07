# Main to train a DNN model

dir_fufiles = "/home/jovyan/DNN_ESANN_dev" # specify directory where function files are located
dir_mofiles = "/home/jovyan/DNN_ESANN_dev" # specify directory where model files are located 

# import packages, libraries, functions to call them later
import ImportAndPrepare_Data
from pytictoc import TicToc
t = TicToc()
from keras.models import load_model

# load data
labels_rand_train, labels_rand_test, an_l_rand_train, an_l_rand_test, an_r_rand_train, an_r_rand_test = ImportAndPrepare_Data.im_and_prep()

# load model
t.tic()
mymodel = load_model(dir_mofiles+"/DNN_model1.h5")
mymodel.summary()
t.toc("loading the model took ")