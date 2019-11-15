# custom metric

from tensorflow.keras import backend as K

def cos_distmet_2D(y_true, y_pred):
    cos_sim = K.sum(y_true*y_pred, axis=1)/(K.sqrt(K.sum(K.square(y_true),axis=1))*K.sqrt(K.sum(K.square(y_pred),axis=1)))
        
    # take the mean across all samples because you have to return one scalar
    cos_dist = 1-cos_sim
    
    return cos_dist