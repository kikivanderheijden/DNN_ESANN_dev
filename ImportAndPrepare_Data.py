
# read and preprocess sounds in the DSRI environment

def im_and_prep():
# define directories, add r before the name because a normal string cannot be used as a path, alternatives are 
# using / or \\ instead of \
    dir_anfiles = "/home/jovyan/Data" # sounds left channel
    
    # import packages and libraries
    import numpy as np
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    import gc # garbage collector
    from pytictoc import TicToc 
    t = TicToc() # create instant of class
    import pickle
    #import seaborn as sns
    
    # set parameters
    testset = 0.20 # size of testset
    
    t.tic()
    # load numpy arrays from disk
    an_l = np.load(dir_anfiles+"/an_l_18000.npy")
    an_r = np.load(dir_anfiles+"/an_r_18000.npy")
    labels = np.load(dir_anfiles+"/labels_18000.npy")
    filenames = pickle.load(open(dir_anfiles+'/listfilenames_18000.p','rb'))
    t.toc("loading the numpy arrays took ")
    
    # shuffle all arrays in the same way
    labels_rand, an_l_rand, an_r_rand, filenames_rand = shuffle(labels, an_l, an_r, filenames, random_state = 0)
    
    # if shuffling is  OK, remove unused variables from memory
    del labels
    del an_l
    del an_r
    gc.collect() # collect garbage to save memory
    
    # create a train and test split
    labels_rand_train, labels_rand_test, an_l_rand_train, an_l_rand_test, an_r_rand_train, an_r_rand_test, filenames_rand_train, filenames_rand_test = train_test_split(labels_rand, an_l_rand, an_r_rand, filenames_rand, test_size = testset, shuffle = False)
    
    # clean memory
    del labels_rand
    del an_l_rand
    del an_r_rand
    gc.collect()
       
    # add a fourth dimension ('channel') to train_an_l and train_an_r which should be 1, this is needed for the input to the DNN
    an_l_rand_train = np.expand_dims(an_l_rand_train,axis = 3)
    an_l_rand_test = np.expand_dims(an_l_rand_test,axis = 3)
    an_r_rand_train = np.expand_dims(an_r_rand_train,axis = 3)
    an_r_rand_test = np.expand_dims(an_r_rand_test,axis = 3)
    
        #save numpy arrays for model evaluation after training
    np.save(dir_anfiles+"/an_l_train_18000.npy",an_l_rand_train)
    np.save(dir_anfiles+"/an_r_train_18000.npy",an_r_rand_train)
    np.save(dir_anfiles+"/an_l_test_18000.npy",an_l_rand_test)
    np.save(dir_anfiles+"/an_r_test_18000.npy",an_r_rand_test)
    np.save(dir_anfiles+"/labels_train_18000.npy",labels_rand_train)
    np.save(dir_anfiles+"/labels_test_18000.npy",labels_rand_test)
    pickle.dump(filenames_rand_train, open(dir_anfiles+'/listfilenames_18000_train.p','wb'))
    pickle.dump(filenames_rand_test, open(dir_anfiles+'/listfilenames_18000_test.p','wb'))

    print("numpy arrays are saved to disk")

    
    print("Shape of training sounds is:", an_l_rand_train.shape)
    print("Shape of training labels is:", labels_rand_train.shape)
    
    print("Shape of test sounds is:", an_l_rand_test.shape)
    print("Shape of test labels is:", labels_rand_test.shape)
    
    #sns.countplot(labels_rand_train[:,0])
    

    
    return labels_rand_train, labels_rand_test, an_l_rand_train, an_l_rand_test, an_r_rand_train, an_r_rand_test
