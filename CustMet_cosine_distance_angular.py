# custom metric

from tensorflow.keras import backend as K

def cos_distmet_2D_angular(y_true, y_pred):
    cos_sim = K.sum(y_true*y_pred, axis=1)/(K.sqrt(K.sum(K.square(y_true),axis=1))*K.sqrt(K.sum(K.square(y_pred),axis=1)))

    cosine_distance_degrees = K.arccos(cos_sim)*180/K.pi
        
    # take the mean across all samples because you have to return one scalar
    
    return cosine_distance_degrees