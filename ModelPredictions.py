# function to generate and save the predictions for the model evaluation

# import necessary libraties
import numpy as np

def generate_model_predictions(model, X_test, modelname, dir_wrfiles):
    
    # generate predictions
    predictions = model.predict(X_test, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=10)
    
    np.save(dir_wrfiles+"/predictions_"+modelname+".npy",predictions)

    return predictions
    
    