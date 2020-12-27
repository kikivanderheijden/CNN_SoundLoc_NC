# script to compare model predictions

#------------------------------------------------------------------------------
# File locations 
#------------------------------------------------------------------------------

# set directories
dirsounds  = '/workspace/notebooks/sounds_small_npy/eval'
dirscripts = '/workspace/notebooks/scripts'
dirfiles   = '/workspace/notebooks/histories'
excelfile  = '/workspace/notebooks/scripts/Overview_ModelPerformance.xlsx'

#------------------------------------------------------------------------------
# import libraries
#------------------------------------------------------------------------------
import numpy as np
import pickle
#import math
import os

os.chdir(dirscripts)
from Evaluation_AnalyzeModelPredictions_Calculations import Evaluation_ModelPredictions_CNN_calculations, Evaluation_ModelPredictions_CNN_calculations_reversalcorrected
from Evaluation_AnalyzeModelPerformanceResults_Calculations import Evaluation_CalculateAveragePerformance
from datetime import datetime

#------------------------------------------------------------------------------
# Specifications
#------------------------------------------------------------------------------

# set azimuth range
azimuthrange = np.arange(0,360,10)
nrevalsounds = 13608 # set number of evaluation sounds

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

# load information about validation set
labels_val =  np.load(dirsounds+"/labels_eval_sounds.npy")
names_val = pickle.load(open(dirsounds+'/listfilenames_eval_sounds.p','rb'))

#------------------------------------------------------------------------------
# Calculation preparations
#------------------------------------------------------------------------------

# create list of file names
os.chdir(dirfiles)
filenames = [x for x in os.listdir(dirfiles) if x.endswith("_predictions.npy")]

# get nr of models
CNN_nrofmodels = 0
for x in range(len(filenames)):
    if 'CNN' in filenames[x]:
        CNN_nrofmodels = CNN_nrofmodels+1
        
# initialize counters
CNN_counter = 0
# initialize matrices - is this the correct dimension?
models_predictions_CNN = np.empty([CNN_nrofmodels,nrevalsounds,2])
filenames_CNN = list(range(CNN_nrofmodels))
for x in range(len(filenames)):
    if 'CNN' in filenames[x]:
        models_predictions_CNN[CNN_counter] = np.load(dirfiles+"/"+filenames[x])
        filenames_CNN[CNN_counter] = filenames[x]
        CNN_counter = CNN_counter+1

        
# checkpoint: do models contain real values or nan? if nan, remove from all lists
nanmodelsCNN = []
for modelcheck in range(len(filenames_CNN)):
    if np.isnan(models_predictions_CNN[modelcheck,0,0]): # check in first value
        nanmodelsCNN = np.append(nanmodelsCNN,modelcheck)
if len(nanmodelsCNN) > 0:# delete the nanmodels everywhere
    models_predictions_CNN = np.delete(models_predictions_CNN, nanmodelsCNN.astype(int),0)
    filenames_CNN = [i for j, i in enumerate(filenames_CNN) if j not in nanmodelsCNN.astype(int)]
    CNN_counter = CNN_counter-len(nanmodelsCNN)

#------------------------------------------------------------------------------
# Calculations for CNNs
#------------------------------------------------------------------------------
# normal calculation
CNN_dict_preds = Evaluation_ModelPredictions_CNN_calculations(names_val, models_predictions_CNN, labels_val, azimuthrange)
# add the filenames to the dictionary 
CNN_dict_preds["filenames_CNN"] = filenames_CNN
# save dictionary to disk for later access
np.save(dirfiles+'/CNN_dict_preds_'+datetime.today().strftime('%Y-%m-%d')+'.npy', CNN_dict_preds)


# corrected for front-back reversals
CNN_dict_preds_revcorr = Evaluation_ModelPredictions_CNN_calculations_reversalcorrected(names_val, models_predictions_CNN, labels_val, azimuthrange)
# add the filenames to the dictionary 
CNN_dict_preds["filenames_CNN"] = filenames_CNN
# save dictionary to disk for later access
np.save(dirfiles+'/CNN_dict_preds_revcorr_'+datetime.today().strftime('%Y-%m-%d')+'.npy', CNN_dict_preds_revcorr)


#------------------------------------------------------------------------------
# Calculations summary/averages
#------------------------------------------------------------------------------

# Calculate values of average performance
CNN_dict_avgperf = Evaluation_CalculateAveragePerformance(CNN_dict_preds, excelfile, filenames_CNN)
# save dictionary to disk for later access
np.save(dirfiles+'/CNN_dict_avgperf_'+datetime.today().strftime('%Y-%m-%d')+'.npy', CNN_dict_avgperf)

