# function to generate and save the predictions for the model evaluation

# import necessary libraties
import numpy as np

def generate_model_predictions(model, X_test, modelname_temp_short, dirfiles, batchs):
    
    # generate predictions
    predictions = model.predict(X_test, batch_size=batchs, verbose=1, steps=None, callbacks=None, max_queue_size=10)
    
    np.save(dirfiles+"\\"+modelname_temp_short+"_predictions.npy",predictions)

    return predictions
    
    