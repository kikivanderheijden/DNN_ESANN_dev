# testing cosine distance function for 2d tensors
from tensorflow.keras import backend as K
# computing the cosine similarity between two 2d vectors
# two fake points
def cos_dist_2D(y_true,y_pred):
   
    # for testing    
    #y_true = np.array([[1,1],[0,1]])
    #y_pred = np.array([[1,1],[0,-1]])
    
    # create some fake data
    #y_true = np.array([[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,-1],[-1,1]])
    #y_pred = np.array([[0,1],[1,0],[0,-1],[-1,0],[-1,-1],[-1,1],[1,1],[1,-1]])
    
       
    #cos_sim = np.sum(y_true*y_pred, axis=1)/(np.sqrt(np.sum(np.square(y_true),axis=1))*np.sqrt(np.sum(np.square(y_pred),axis=1)))
    cos_sim = K.sum(y_true*y_pred, axis=1)/(K.sqrt(K.sum(K.square(y_true),axis=1))*K.sqrt(K.sum(K.square(y_pred),axis=1)))
        
    # take the mean across all samples because you have to return one scalar
    cos_dist = 1-cos_sim
    
    return cos_dist


# OLD CODE
# =============================================================================
#     # retrieve the number of pairs
#     nrbatches = np.shape(y_true)
#     nrbatches = nrbatches[0]
#     
#     cos_sim = np.empty(nrbatches)
#     for x in range(nrbatches):
#         cos_sim[x] = np.dot(y_true[x],y_pred[x])/(np.sqrt(np.sum(np.square(y_true[x])))*np.sqrt(np.sum(np.square(y_pred[x]))))
# 
# =============================================================================
