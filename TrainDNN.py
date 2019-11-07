# Main to train a DNN model

dir_fufiles = "/home/jovyan/DNN_ESANN_dev" # specify directory where function files are located
dir_mofiles = "/home/jovyan/DNN_ESANN_dev" # specify directory where model files are located 

#dir_mofiles = r"C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning"

# import packages, libraries, functions to call them later
#import ImportAndPrepare_Data
from pytictoc import TicToc
t = TicToc()
from tensorflow.keras.models import load_model
#from tensorflow.keras.models import model_from_json
from tensorflow.keras.initializers import glorot_uniform

# load data
#labels_rand_train, labels_rand_test, an_l_rand_train, an_l_rand_test, an_r_rand_train, an_r_rand_test = ImportAndPrepare_Data.im_and_prep()

# =============================================================================
# t.tic()
# json_file = open(dir_mofiles+'/DNN_model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.summary()
# t.toc("loading the model took")
# =============================================================================

# load model
t.tic()
mymodel = load_model(dir_mofiles+"/DNN_model1.h5",custom_objects={'GlorotUniform': glorot_uniform()})
mymodel.summary()
t.toc("loading the model took ")