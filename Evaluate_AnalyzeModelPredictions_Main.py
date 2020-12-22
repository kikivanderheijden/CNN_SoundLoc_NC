# script to compare model predictions

#------------------------------------------------------------------------------
# File locations 
#------------------------------------------------------------------------------

# set directories
dirfiles = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\CNN_SoundLoc_development_GPU\EvaluationFiles'
dirscripts = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\CNN_SoundLoc_development_GPU'
excelfile = r'C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning\DNN_HumanSoundLoc\DNN_modelspace_overview_performance.xlsx'

#------------------------------------------------------------------------------
# import libraries
#------------------------------------------------------------------------------
import numpy as np
import pickle
import matplotlib.pyplot as plt
#import math
import os
import statsmodels.api as sm

os.chdir(dirscripts)
from Evaluation_AnalyzeModelPredictions_Calculations import Evaluation_ModelPredictions_CNN_calculations, Evaluation_ModelPredictions_CNN_calculations_reversalcorrected
from Evaluation_AnalyzeModelPerformanceResults_Calculations import Evaluation_CalculateAveragePerformance
from datetime import datetime

#------------------------------------------------------------------------------
# Specifications
#------------------------------------------------------------------------------

# set azimuth range
azimuthrange = np.arange(0,360,10)

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

# load information about validation set
labels_val =  np.load(dirfiles+"/labels_eval_sounds.npy")
names_val = pickle.load(open(dirfiles+'/listfilenames_eval_sounds.p','rb'))
labels_val_RNN =  np.load(dirfiles+"/labels_eval_sounds.npy")
names_val_RNN = pickle.load(open(dirfiles+'/listfilenames_eval_sounds.p','rb'))

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
# initialize matrices !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! change dimension of 3600 here into correct dimension !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
models_predictions_CNN = np.empty([CNN_nrofmodels,3600,2])
filenames_CNN = list(range(CNN_nrofmodels))
for x in range(len(filenames)):
    if 'CNN' in filenames[x]:
        models_predictions_CNN[CNN_counter] = np.load(dirfiles+"\\"+filenames[x])
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

#------------------------------------------------------------------------------
# Statistics
#------------------------------------------------------------------------------ 

## Effect model complexity, ordinary least squares regression using statsmodel
# for CNNs
X = CNN_dict_avgperf["CNN_mse_nrparams"]
y = CNN_dict_avgperf["CNN_mse_scoremse"]
X2 = sm.add_constant(X) # add the constant
mse_scoremse_est = sm.OLS(y,X2)
mse_scoremse_est2 = mse_scoremse_est.fit()
mse_scoremse_est2_params = mse_scoremse_est.fit().params
print(mse_scoremse_est2.summary())

X = CNN_dict_avgperf["CNN_ad_nrparams"]
y = CNN_dict_avgperf["CNN_ad_scoremse"]
X2 = sm.add_constant(X) # add the constant
ad_scoremse_est = sm.OLS(y,X2)
ad_scoremse_est2 = ad_scoremse_est.fit()
print(ad_scoremse_est2.summary())

X = CNN_dict_avgperf["CNN_mse_nrparams"]
y = CNN_dict_avgperf["CNN_mse_scoread"]
X2 = sm.add_constant(X) # add the constant
mse_scoread_est = sm.OLS(y,X2)
mse_scoread_est2 = mse_scoread_est.fit()
mse_scoread_est2_params = mse_scoread_est.fit().params
print(mse_scoread_est2.summary())

X = CNN_dict_avgperf["CNN_ad_nrparams"]
y = CNN_dict_avgperf["CNN_ad_scoread"]
X2 = sm.add_constant(X) # add the constant
ad_scoread_est = sm.OLS(y,X2)
ad_scoread_est2 = ad_scoread_est.fit()
ad_scoread_est2_params = ad_scoread_est.fit().params
print(ad_scoread_est2.summary())


## Test whether training loss also predicts the other loss, ordinary least squares regression
# first, trained on MSE loss, does MSE loss predict AD loss?
X = CNN_dict_avgperf["CNN_mse_scoremse"]
y = CNN_dict_avgperf["CNN_mse_scoread"]
X2 = sm.add_constant(X) # add the constant
cnn_mse_adest = sm.OLS(y,X2)
cnn_mse_adest2 = cnn_mse_adest.fit()
cnn_mse_adest2_params = cnn_mse_adest.fit().params
print(cnn_mse_adest2.summary())

X = CNN_dict_avgperf["CNN_ad_scoread"]
y = CNN_dict_avgperf["CNN_ad_scoremse"]
X2 = sm.add_constant(X) # add the constant
cnn_ad_mseest = sm.OLS(y,X2)
cnn_ad_mseest2 = cnn_ad_mseest.fit()
cnn_ad_mseest2_params = cnn_ad_mseest.fit().params
print(cnn_ad_mseest2.summary())

#------------------------------------------------------------------------------
# Create figures
#------------------------------------------------------------------------------

## Figure of average model performance as a function of model complexity
# first create scatterplots of model performance as a function of model complexity, in terms of MSE loss
darkblue = '#0026b3'
plt.figure()
plt.scatter(CNN_dict_avgperf["CNN_mse_nrparams"],CNN_dict_avgperf["CNN_mse_scoremse"], s = 100, marker = "^", c = darkblue) 
plt.scatter(CNN_dict_avgperf["CNN_ad_nrparams"],CNN_dict_avgperf["CNN_ad_scoremse"], s = 100, marker = "^", edgecolors = darkblue, facecolors = 'none', linewidth = 2) 
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
plt.legend(['CNN | MSE loss', 'CNN | AD loss'], fontsize = 10)
plt.xlim(0, 350000)
plt.ylim(0, 0.25)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Number of model parameters', fontsize = 12, fontweight = 'bold')
plt.ylabel('MSE score', fontsize = 12, fontweight = 'bold')
# plot model fit
plt.plot(CNN_dict_avgperf["CNN_mse_nrparams"],(mse_scoremse_est2_params[1]* CNN_dict_avgperf["CNN_mse_nrparams"])+ mse_scoremse_est2_params[0],'-', c = 'darkblue')
plt.savefig(dirfiles+'/Scatterplot_MSE_PerformanceAsFunctionOfModelParams_Python.eps')
 
 # first create scatterplots of model performance as a function of model complexity, in terms of AD loss
darkblue = '#0026b3'
plt.figure()
plt.scatter(CNN_dict_avgperf["CNN_mse_nrparams"],CNN_dict_avgperf["CNN_mse_scoread"], s = 100, marker = "^", c = darkblue) 
plt.scatter(CNN_dict_avgperf["CNN_ad_nrparams"],CNN_dict_avgperf["CNN_ad_scoread"], s = 100, marker = "^", edgecolors = darkblue, facecolors = 'none', linewidth = 2) 
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
#plt.legend(['CNN | MSE loss', 'CNN | AD loss','RNN | MSE loss', 'RNN | AD loss'], fontsize = 10)
plt.xlim(0, 350000)
plt.ylim(0, 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Number of model parameters', fontsize = 12, fontweight = 'bold')
plt.ylabel('AD score (degrees)', fontsize = 12, fontweight = 'bold')
plt.plot(CNN_dict_avgperf["CNN_mse_nrparams"],(mse_scoread_est2_params[1]* CNN_dict_avgperf["CNN_mse_nrparams"])+ mse_scoread_est2_params[0],'-', c = 'darkblue')
plt.plot(CNN_dict_avgperf["CNN_ad_nrparams"],(ad_scoread_est2_params[1]* CNN_dict_avgperf["CNN_ad_nrparams"])+ ad_scoread_est2_params[0],':', c = 'darkblue', linewidth = 1)
plt.savefig(dirfiles+'/Scatterplot_AD_PerformanceAsFunctionOfModelParams_Python.eps')

## Boxplots of performance per model class
# MSE score
plt.figure()
boxprops1 = dict(linewidth = 2, color='darkblue', facecolor = 'darkblue')
boxprops2 = dict(linewidth = 2,color='darkblue')
boxprops3 = dict(linewidth = 2,color= 'black', facecolor = 'black')
boxprops4 = dict(linewidth = 2)
medianprops1 = dict(linewidth = 2)
whiskerprops1 = dict(linewidth = 2,color = 'darkblue')
whiskerprops2 = dict(linewidth = 2,color = 'black')
capprops1 = dict(linewidth = 2, color='darkblue')
capprops2 = dict(linewidth = 2, color='black')
flierprops1 = dict(markeredgewidth = 2, marker = '+', markeredgecolor='darkblue')
plt.boxplot(CNN_dict_avgperf["CNN_mse_scoremse"], patch_artist = True, boxprops = boxprops1, whiskerprops = whiskerprops1, capprops = capprops1, medianprops = medianprops1, positions = [1]) 
plt.boxplot(CNN_dict_avgperf["CNN_ad_scoremse"], boxprops = boxprops2, whiskerprops = whiskerprops1, flierprops = flierprops1, capprops = capprops1, medianprops = medianprops1, positions = [1.5]) 
plt.ylim(0, 0.25)
plt.ylabel('MSE score', fontsize = 12, fontweight = 'bold')
plt.xlabel('Model class', fontsize = 12, fontweight = 'bold')
plt.grid(color = 'k', linestyle = ':', linewidth = .5, alpha = .5, axis = 'y') 
plt.xticks([1, 1.5, 2, 2.5], ['CNN\nMSE loss', 'CNN\nAD loss'])
plt.savefig(dirfiles+'/Boxplot_MSE_PerformanceAverageModelClass_Python.eps')
# AD score
plt.figure()
boxprops1 = dict(linewidth = 2, color='darkblue', facecolor = 'darkblue')
boxprops2 = dict(linewidth = 2,color='darkblue')
boxprops3 = dict(linewidth = 2,color= 'black', facecolor = 'black')
boxprops4 = dict(linewidth = 2)
medianprops1 = dict(linewidth = 2)
whiskerprops1 = dict(linewidth = 2,color = 'darkblue')
whiskerprops2 = dict(linewidth = 2,color = 'black')
capprops1 = dict(linewidth = 2, color='darkblue')
capprops2 = dict(linewidth = 2, color='black')
flierprops1 = dict(markeredgewidth = 2, marker = '+', markeredgecolor='darkblue')
plt.boxplot(CNN_dict_avgperf["CNN_mse_scoread"], patch_artist = True, boxprops = boxprops1, whiskerprops = whiskerprops1, capprops = capprops1, medianprops = medianprops1, positions = [1]) 
plt.boxplot(CNN_dict_avgperf["CNN_ad_scoread"], boxprops = boxprops2, whiskerprops = whiskerprops1, flierprops = flierprops1, capprops = capprops1, medianprops = medianprops1, positions = [1.5]) 
plt.ylim(0, 12)
plt.ylabel('AD score (degrees)', fontsize = 12, fontweight = 'bold')
plt.xlabel('Model class', fontsize = 12, fontweight = 'bold')
plt.grid(color = 'k', linestyle = ':', linewidth = .5, alpha = .5, axis = 'y') 
plt.xticks([1, 1.5, 2, 2.5], ['CNN\nMSE loss', 'CNN\nAD loss'])
plt.savefig(dirfiles+'/Boxplot_AD_PerformanceAverageModelClass_Python.eps')

## Scatterplot of MSE score plotted against AD score for CNN and RNN MSE models
plt.figure()
plt.scatter(CNN_dict_avgperf["CNN_mse_scoremse"],CNN_dict_avgperf["CNN_mse_scoread"], s = 100, marker = "^", c = darkblue) 
plt.xlim(0, 0.06)
plt.ylim(0, 12)
plt.legend(['CNN | MSE loss'], fontsize = 10, loc = 'upper left')
plt.ylabel('AD score (degrees)', fontsize = 12, fontweight = 'bold')
plt.xlabel('MSE score', fontsize = 12, fontweight = 'bold')
plt.plot(CNN_dict_avgperf["CNN_mse_scoremse"],(CNN_dict_avgperf["CNN_mse_scoremse"]*cnn_mse_adest2_params[1])+cnn_mse_adest2_params[0],'-', c = 'darkblue')
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
plt.savefig(dirfiles+'/Scatterplot_MSEagainstADscore_for_CNNwithMSE_Python.eps')

# scatterplot of AD score againnst MSE score for CNN and RNN AD models    
plt.figure()
plt.scatter(CNN_dict_avgperf["CNN_ad_scoread"],CNN_dict_avgperf["CNN_ad_scoremse"], s = 100, marker = "^", edgecolors = darkblue, facecolors = 'none', linewidth = 2) 
plt.xlim(0, 12)
plt.ylim(0, 0.2)
plt.ylabel('MSE score', fontsize = 12, fontweight = 'bold')
plt.xlabel('AD score (degrees)', fontsize = 12, fontweight = 'bold')
plt.legend(['CNN | AD loss'], fontsize = 10, loc = 'upper left')
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
plt.savefig(dirfiles+'/Scatterplot_MSEagainstADscore_for_CNNwithAD_Python.eps')
  
## Plot of the effect of kernel size on MSE or AD score
# prepare data
# first model type trained with MSE
CNN_S_L4_K323264128_mse_1 = CNN_dict_avgperf['CNN_mse_scoremse'][1:5] # get values KS 1x3 until 1x9
CNN_S_L4_K323264128_mse_1 = np.reshape(CNN_S_L4_K323264128_mse_1,len(CNN_S_L4_K323264128_mse_1,)) 
CNN_S_L4_K323264128_mse_2 = CNN_dict_avgperf['CNN_mse_scoremse'][0] # get value KS 1x11
CNN_S_L4_K323264128_mse_2 = np.reshape(CNN_S_L4_K323264128_mse_2,len(CNN_S_L4_K323264128_mse_2,))
CNN_S_L4_K323264128_mse = np.concatenate((CNN_S_L4_K323264128_mse_1,CNN_S_L4_K323264128_mse_2))
# second model type trained with MSE
CNN_S_L4_K32326464_mse_1 = CNN_dict_avgperf['CNN_mse_scoremse'][6:10] # get values KS 1x3 until 1x9
CNN_S_L4_K32326464_mse_1 = np.reshape(CNN_S_L4_K32326464_mse_1,len(CNN_S_L4_K32326464_mse_1,))
CNN_S_L4_K32326464_mse_2 = CNN_dict_avgperf['CNN_mse_scoremse'][5] # get values KS 1x3 until 1x9
CNN_S_L4_K32326464_mse_2 = np.reshape(CNN_S_L4_K32326464_mse_2,len(CNN_S_L4_K32326464_mse_2,))
CNN_S_L4_K32326464_mse = np.concatenate((CNN_S_L4_K32326464_mse_1,CNN_S_L4_K32326464_mse_2))
# third model type trained with MSE
CNN_S_L4_K646464128_mse_1 = CNN_dict_avgperf['CNN_mse_scoremse'][11:]  # get values KS 1x3 until 1x9
CNN_S_L4_K646464128_mse_1 = np.reshape(CNN_S_L4_K646464128_mse_1,len(CNN_S_L4_K646464128_mse_1,))
CNN_S_L4_K646464128_mse_2 = CNN_dict_avgperf['CNN_mse_scoremse'][10]  # get values KS 1x3 until 1x9
CNN_S_L4_K646464128_mse_2 = np.reshape(CNN_S_L4_K646464128_mse_2,len(CNN_S_L4_K646464128_mse_2,))
CNN_S_L4_K646464128_mse = np.concatenate((CNN_S_L4_K646464128_mse_1,CNN_S_L4_K646464128_mse_2))
# first model type trained with AD
CNN_S_L4_K323264128_ad_1 = CNN_dict_avgperf['CNN_ad_scoread'][1:5]# get values KS 1x3 until 1x9
CNN_S_L4_K323264128_ad_1 = np.reshape(CNN_S_L4_K323264128_ad_1, len(CNN_S_L4_K323264128_ad_1,))
CNN_S_L4_K323264128_ad_2 = CNN_dict_avgperf['CNN_ad_scoread'][0]# get values KS 1x3 until 1x9
CNN_S_L4_K323264128_ad_2 = np.reshape(CNN_S_L4_K323264128_ad_2, len(CNN_S_L4_K323264128_ad_2,))
CNN_S_L4_K323264128_ad = np.concatenate((CNN_S_L4_K323264128_ad_1,CNN_S_L4_K323264128_ad_2))
# second model type trained with AD
CNN_S_L4_K32326464_ad_1 = CNN_dict_avgperf['CNN_ad_scoread'][6:10]# get values KS 1x3 until 1x9
CNN_S_L4_K32326464_ad_1 = np.reshape(CNN_S_L4_K32326464_ad_1,len(CNN_S_L4_K32326464_ad_1,))
CNN_S_L4_K32326464_ad_2 = CNN_dict_avgperf['CNN_ad_scoread'][5]# get values KS 1x3 until 1x9
CNN_S_L4_K32326464_ad_2 = np.reshape(CNN_S_L4_K32326464_ad_2,len(CNN_S_L4_K32326464_ad_2,))
CNN_S_L4_K32326464_ad = np.concatenate((CNN_S_L4_K32326464_ad_1,CNN_S_L4_K32326464_ad_2))
# third model type trained with AD
CNN_S_L4_K646464128_ad_1 = CNN_dict_avgperf['CNN_ad_scoread'][11:]  # get values KS 1x3 until 1x9
CNN_S_L4_K646464128_ad_1 = np.reshape(CNN_S_L4_K646464128_ad_1,len(CNN_S_L4_K646464128_ad_1,))
CNN_S_L4_K646464128_ad_2 = CNN_dict_avgperf['CNN_ad_scoread'][10]  # get values KS 1x3 until 1x9
CNN_S_L4_K646464128_ad_2 = np.reshape(CNN_S_L4_K646464128_ad_2,len(CNN_S_L4_K646464128_ad_2,))
CNN_S_L4_K646464128_ad = np.concatenate((CNN_S_L4_K646464128_ad_1,CNN_S_L4_K646464128_ad_2))

# plot the data
plt.figure()
plt.plot(CNN_S_L4_K32326464_mse, linestyle = ':', marker = 'o', color = 'black')
plt.plot(CNN_S_L4_K323264128_mse, linestyle = '--', marker = 'o', color = 'black')
plt.plot(CNN_S_L4_K646464128_mse, marker = 'o', color = 'black')
plotavg_mse = np.mean(np.concatenate(([CNN_S_L4_K323264128_mse],[CNN_S_L4_K32326464_mse],[CNN_S_L4_K646464128_mse])),axis = 0)
plt.plot(plotavg_mse, marker = 'o', markersize = 7, color = 'red', linewidth = 3)
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
plt.ylim(0,.05)
plt.xlabel('Kernel size', fontsize = 12, fontweight = 'bold')
plt.xticks([0,1,2,3,4],['1x3','1x5','1x7','1x9','1x11'])
plt.ylabel('MSE', fontsize = 12, fontweight = 'bold')
plt.legend(['Kernels per layer: 32-32-64-64', 'Kernels per layer: 32-32-64-128', 'Kernels per layer: 64-64-64-128', 'Average across model types'], fontsize = 10, loc = 'lower right')
plt.savefig(dirfiles+'/LinePlot_EffectKernelSize_CNNmodels_withMSE_Python.eps')

plt.figure()
plt.plot(CNN_S_L4_K32326464_ad, linestyle = ':', marker = 'o', color = 'black')
plt.plot(CNN_S_L4_K323264128_ad, linestyle = '--', marker = 'o', color = 'black')
plt.plot(CNN_S_L4_K646464128_ad, marker = 'o', color = 'black')
plotavg_ad = np.mean(np.concatenate(([CNN_S_L4_K323264128_ad],[CNN_S_L4_K32326464_ad],[CNN_S_L4_K646464128_ad])),axis = 0)
plt.plot(plotavg_ad, marker = 'o', markersize = 7, color = 'red', linewidth = 3)
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
plt.ylim(0,.05)
plt.xlabel('Kernel size', fontsize = 12, fontweight = 'bold')
plt.xticks([0,1,2,3,4],['1x3','1x5','1x7','1x9','1x11'])
plt.ylabel('AD (degrees)', fontsize = 12, fontweight = 'bold')
plt.ylim(0,12)
plt.legend(['Kernels per layer: 32-32-64-64', 'Kernels per layer: 32-32-64-128', 'Kernels per layer: 64-64-64-128', 'Average across model types'], fontsize = 10, loc = 'lower right')
plt.savefig(dirfiles+'/LinePlot_EffectKernelSize_CNNmodels_withAD_Python.eps')


## Boxplots of percentage of front-back reversals per model class
plt.figure()
boxprops1 = dict(linewidth = 2, color='darkblue', facecolor = 'darkblue')
boxprops2 = dict(linewidth = 2,color='darkblue')
boxprops3 = dict(linewidth = 2,color= 'black', facecolor = 'black')
boxprops4 = dict(linewidth = 2)
medianprops1 = dict(linewidth = 2)
whiskerprops1 = dict(linewidth = 2,color = 'darkblue')
whiskerprops2 = dict(linewidth = 2,color = 'black')
capprops1 = dict(linewidth = 2, color='darkblue')
capprops2 = dict(linewidth = 2, color='black')
flierprops1 = dict(markeredgewidth = 2, marker = '+', markeredgecolor='darkblue')
flierprops2 = dict(markeredgewidth = 2, marker = '+', markeredgecolor='black')
plt.boxplot(CNN_dict_preds_revcorr['reversals_percentage'][1::2], patch_artist = True, boxprops = boxprops1, whiskerprops = whiskerprops1, flierprops = flierprops1, capprops = capprops1, medianprops = medianprops1, positions = [1]) 
plt.boxplot(CNN_dict_preds_revcorr['reversals_percentage'][0::2], boxprops = boxprops2, whiskerprops = whiskerprops1, flierprops = flierprops1, capprops = capprops1, medianprops = medianprops1, positions = [1.5]) 
plt.ylim(0, 0.025)
plt.ylabel('Front-back reversals (%)', fontsize = 12, fontweight = 'bold')
plt.xlabel('Model class', fontsize = 12, fontweight = 'bold')
plt.grid(color = 'k', linestyle = ':', linewidth = .5, alpha = .5, axis = 'y') 
plt.xticks([1, 1.5, 2, 2.5], ['CNN\nMSE loss', 'CNN\nAD loss'])
plt.savefig(dirfiles+'/Boxplot_FrontBackReversals_perModelClass_Python.eps')

## confusion matrices of predictions
# CNN models original data
# cm_tarlocs = np.arange(0,370,10)
cm_tarlocs = [0,5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195,205,215,225,235,245,255,265,275,285,295,305,315,325,335,345,355,360]
CNN_cm_preds = CNN_dict_preds['models_predictions_CNN_degs_azloc']
CNN_cm_preds_hist = np.zeros((len(CNN_cm_preds),len(CNN_cm_preds[0,:,:]),len(cm_tarlocs)-1)) # note that in histogram the last one is included in the one but last
for w in range(len(CNN_cm_preds)):
    cm_preds = CNN_cm_preds[w,:,:]
    # cycle through the location to get the matrix
    for x in range(len(cm_preds)):
        tempdata = np.histogram(cm_preds[x], cm_tarlocs)
        CNN_cm_preds_hist[w,x,:] = tempdata[0]
# now sum the first and last column
CNN_cm_preds_hist_original = CNN_cm_preds_hist # save original so you do not lose it         
zerobin = CNN_cm_preds_hist[:,:,0]+CNN_cm_preds_hist[:,:,len(CNN_cm_preds_hist[0,0,:])-1] # compute zerobin from -5 to 5 degrees
CNN_cm_preds_hist = np.concatenate((np.expand_dims(zerobin,axis=2), CNN_cm_preds_hist[:,:,1:-1]), axis = 2)
# get frequencies instead of counts
CNN_cm_preds_hist = CNN_cm_preds_hist/len(CNN_cm_preds[0,0,:])
# shift the matrices  per axis (first rows, then columnns)
CNN_cm_preds_hist_shift = np.roll(CNN_cm_preds_hist,18,axis = 2)
CNN_cm_preds_hist_shift = np.roll(CNN_cm_preds_hist_shift,18,axis = 1)
# create confusion matrices
for x in range(len(CNN_cm_preds_hist_shift)):
    tempname = filenames_CNN[x]
    tempnameshort = tempname[9:-16]
    plt.matshow(np.flipud(CNN_cm_preds_hist_shift[x,:,:]), cmap = 'Spectral_r', vmax = 1)
    plt.gca().xaxis.tick_bottom()
    plt.xticks([-.5,9,19,28,35.5],['180','270','0','90','170'], fontsize = 12)
    plt.yticks([-.5,7,16,26,35.5],['170','90','0','270','180'], fontsize = 12)
    plt.ylabel('Target location', fontsize = 12)
    plt.xlabel('Predicted location', fontsize = 12)
    plt.colorbar()
    plt.savefig(dirfiles+'/CNN_'+tempnameshort+'_ConfusionMatrixPredictions.eps')
    plt.close('all')
# CNN models reversal corrected
cm_tarlocs = [0,5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195,205,215,225,235,245,255,265,275,285,295,305,315,325,335,345,355,360]
CNN_cm_preds = CNN_dict_preds_revcorr['models_predictions_CNN_degs_azloc_corrected']
CNN_cm_preds_hist = np.zeros((len(CNN_cm_preds),len(CNN_cm_preds[0,:,:]),len(cm_tarlocs)-1)) # note that in histogram the last one is included in the one but last
for w in range(len(CNN_cm_preds)):
    cm_preds = CNN_cm_preds[w,:,:]
    # cycle through the location to get the matrix
    for x in range(len(cm_preds)):
        tempdata = np.histogram(cm_preds[x], cm_tarlocs)
        CNN_cm_preds_hist[w,x,:] = tempdata[0]
# now sum the first and last column
CNN_cm_preds_hist_original = CNN_cm_preds_hist # save original so you do not lose it         
zerobin = CNN_cm_preds_hist[:,:,0]+CNN_cm_preds_hist[:,:,len(CNN_cm_preds_hist[0,0,:])-1] # compute zerobin from -5 to 5 degrees
CNN_cm_preds_hist = np.concatenate((np.expand_dims(zerobin,axis=2), CNN_cm_preds_hist[:,:,1:-1]), axis = 2)
# get frequencies instead of counts
CNN_cm_preds_hist = CNN_cm_preds_hist/len(CNN_cm_preds[0,0,:])
# shift the matrices  per axis (first rows, then columnns)
CNN_cm_preds_hist_shift = np.roll(CNN_cm_preds_hist,18,axis = 2)
CNN_cm_preds_hist_shift = np.roll(CNN_cm_preds_hist_shift,18,axis = 1)
# create confusion matrices
for x in range(len(CNN_cm_preds_hist_shift)):
    tempname = filenames_CNN[x]
    tempnameshort = tempname[9:-16]
    plt.matshow(np.flipud(CNN_cm_preds_hist_shift[x,:,:]), cmap = 'Spectral_r', vmax = 1)
    plt.gca().xaxis.tick_bottom()
    plt.xticks([-.5,9,19,28,35.5],['180','270','0','90','170'], fontsize = 12)
    plt.yticks([-.5,7,16,26,35.5],['170','90','0','270','180'], fontsize = 12)
    plt.ylabel('Target location', fontsize = 12)
    plt.xlabel('Predicted location', fontsize = 12)
    plt.colorbar()
    plt.savefig(dirfiles+'/CNN_'+tempnameshort+'_ConfusionMatrixPredictions_revcorrected.eps')
    plt.close('all')

# RNN models original
cm_tarlocs = [0,5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195,205,215,225,235,245,255,265,275,285,295,305,315,325,335,345,355,360]
RNN_cm_preds = RNN_dict_preds['models_predictions_RNN_degs_azloc']
RNN_cm_preds_hist = np.zeros((len(RNN_cm_preds),len(RNN_cm_preds[0,:,:]),len(cm_tarlocs)-1)) # note that in histogram the last one is included in the one but last
for w in range(len(RNN_cm_preds)):
    cm_preds = RNN_cm_preds[w,:,:]
    # cycle through the location to get the matrix
    for x in range(len(cm_preds)):
        tempdata = np.histogram(cm_preds[x], cm_tarlocs)
        RNN_cm_preds_hist[w,x,:] = tempdata[0]
# now sum the first and last column
RNN_cm_preds_hist_original = RNN_cm_preds_hist # save original so you do not lose it         
zerobin = RNN_cm_preds_hist[:,:,0]+RNN_cm_preds_hist[:,:,len(RNN_cm_preds_hist[0,0,:])-1] # compute zerobin from -5 to 5 degrees
RNN_cm_preds_hist = np.concatenate((np.expand_dims(zerobin,axis=2), RNN_cm_preds_hist[:,:,1:-1]), axis = 2)
# get frequencies instead of counts
RNN_cm_preds_hist = RNN_cm_preds_hist/len(RNN_cm_preds[0,0,:])
# shift the matrices  per axis (first rows, then columnns)
RNN_cm_preds_hist_shift = np.roll(RNN_cm_preds_hist,18,axis = 2)
RNN_cm_preds_hist_shift = np.roll(RNN_cm_preds_hist_shift,18,axis = 1)
# create confusion matrices
for x in range(len(RNN_cm_preds_hist_shift)):
    tempname = filenames_RNN[x]
    tempnameshort = tempname[8:-16]
    plt.matshow(np.flipud(RNN_cm_preds_hist_shift[x,:,:]), cmap = 'Spectral_r', vmax = 1)
    plt.gca().xaxis.tick_bottom()
    plt.xticks([-.5,9,19,28,35.5],['180','270','0','90','170'], fontsize = 12)
    plt.yticks([-.5,7,16,26,35.5],['170','90','0','270','180'], fontsize = 12)
    plt.ylabel('Target location', fontsize = 12)
    plt.xlabel('Predicted location', fontsize = 12)
    plt.colorbar()
    plt.savefig(dirfiles+'/RNN_'+tempnameshort+'ConfusionMatrixPredictions.eps')
    plt.close('all')
# RNN models reversal corrected
cm_tarlocs = [0,5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195,205,215,225,235,245,255,265,275,285,295,305,315,325,335,345,355,360]
RNN_cm_preds = RNN_dict_preds_revcorr['models_predictions_RNN_degs_azloc_corrected']
RNN_cm_preds_hist = np.zeros((len(RNN_cm_preds),len(RNN_cm_preds[0,:,:]),len(cm_tarlocs)-1)) # note that in histogram the last one is included in the one but last
for w in range(len(RNN_cm_preds)):
    cm_preds = RNN_cm_preds[w,:,:]
    # cycle through the location to get the matrix
    for x in range(len(cm_preds)):
        tempdata = np.histogram(cm_preds[x], cm_tarlocs)
        RNN_cm_preds_hist[w,x,:] = tempdata[0]
# now sum the first and last column
RNN_cm_preds_hist_original = RNN_cm_preds_hist # save original so you do not lose it         
zerobin = RNN_cm_preds_hist[:,:,0]+RNN_cm_preds_hist[:,:,len(RNN_cm_preds_hist[0,0,:])-1] # compute zerobin from -5 to 5 degrees
RNN_cm_preds_hist = np.concatenate((np.expand_dims(zerobin,axis=2), RNN_cm_preds_hist[:,:,1:-1]), axis = 2)
# get frequencies instead of counts
RNN_cm_preds_hist = RNN_cm_preds_hist/len(RNN_cm_preds[0,0,:])
# shift the matrices  per axis (first rows, then columnns)
RNN_cm_preds_hist_shift = np.roll(RNN_cm_preds_hist,18,axis = 2)
RNN_cm_preds_hist_shift = np.roll(RNN_cm_preds_hist_shift,18,axis = 1)
# create confusion matrices
for x in range(len(RNN_cm_preds_hist_shift)):
    tempname = filenames_RNN[x]
    tempnameshort = tempname[8:-16]
    plt.matshow(np.flipud(RNN_cm_preds_hist_shift[x,:,:]), cmap = 'Spectral_r', vmax = 1)
    plt.gca().xaxis.tick_bottom()
    plt.xticks([-.5,9,19,28,35.5],['180','270','0','90','170'], fontsize = 12)
    plt.yticks([-.5,7,16,26,35.5],['170','90','0','270','180'], fontsize = 12)
    plt.ylabel('Target location', fontsize = 12)
    plt.xlabel('Predicted location', fontsize = 12)
    plt.colorbar()
    plt.savefig(dirfiles+'/RNN_'+tempnameshort+'ConfusionMatrixPredictions_revcorrected.eps')
    plt.close('all')

# for humans
humanpreds = np.array(((.75,.12,.015,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.015,.12),\
                       (.12,.75,.12,.015,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.015),\
                       (.015,.12,.75,.12,.015,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,.075,.8,.075,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.075,.8,.075,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.005,.9,.005,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.005,.9,.005,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.005,.9,.005,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.075,.8,.075,0,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.075,.8,.075,0,0,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0,0,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.07,.07,.12,.5,.12,.07,.07,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.03,.12,.6,.12,.03,0),\
                       (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.015,.12,.75,.12,.015),\
                       (.015,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.015,.12,.75,.12),\
                       (.12,.015,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.015,.12,.75),\
                       ))
plt.matshow(np.flipud(humanpreds), cmap = 'Spectral_r', vmax = 1)
plt.gca().xaxis.tick_bottom()
plt.xticks([-.5,9,19,28,35.5],['180','270','0','90','170'], fontsize = 12)
plt.yticks([-.5,7,16,26,35.5],['170','90','0','270','180'], fontsize = 12)
plt.ylabel('Target location', fontsize = 12)
plt.xlabel('Predicted location', fontsize = 12)
plt.colorbar()
plt.savefig(dirfiles+'/Human_ConfusionMatrixPredictions.eps')
plt.close('all')

## polar plot of the error as a function of azimuth per model
# loop through all CNN models and save the plots with the name
# for MSE
for x in range(len(filenames_CNN)):
    tempname = filenames_CNN[x]
    tempname = tempname[:-16]
    CNN_mean_mse_az = CNN_dict_preds['mean_mse_az'][x,]
    CNN_mean_mse_az_revcorr = CNN_dict_preds_revcorr['mean_mse_az_corrected'][x,]
    radii_mse_1 = CNN_mean_mse_az
    radii_mse_2 = np.zeros(1)
    radii_mse_2[0] = CNN_mean_mse_az[0]
    radii_mse = np.concatenate((radii_mse_1, radii_mse_2)) # you have to add the last point at the end of the array to close the circle
    radii_msecor_1 = CNN_mean_mse_az_revcorr
    radii_msecor_2 = np.zeros(1)
    radii_msecor_2[0] = CNN_mean_mse_az_revcorr[0]
    radii_msecor = np.concatenate((radii_msecor_1, radii_msecor_2)) # you have to add the last point at the end of the array to close the circle
    plt.figure(figsize = (3,3))
    ax = plt.subplot(111, projection='polar')
    theta_1 = azimuthrange
    theta_2 = np.zeros(1)
    theta_2[0] = azimuthrange[0]
    theta = np.concatenate((theta_1, theta_2))
    ax.plot(np.radians(theta),radii_mse, color = 'black', linewidth = 2)
    ax.plot(np.radians(theta),radii_msecor, linestyle = '--', color = 'red', linewidth = 1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if 'MSE' in tempname:
        ax.set_ylim(0,.30)
        ax.set_yticks(np.arange(0,.30,.1))
    elif 'AD' in tempname:
        ax.set_ylim(0,.30)
        ax.set_yticks(np.arange(0,.30,.1))
    #ax.legend(('Raw', 'Reversal Corrected'), fontsize = 10,bbox_to_anchor=(1,.1)) # this latter thing is to put the legend in the correct location
    plt.grid(linestyle = ':', linewidth = 1)
    plt.savefig(dirfiles+'/PolarPlot_MSE_'+tempname+'.eps')
    plt.close('all')
for x in range(len(filenames_RNN)):
    tempname = filenames_RNN[x]
    tempname = tempname[2:-16]
    RNN_mean_mse_az = RNN_dict_preds['mean_mse_az_RNN'][x,]
    RNN_mean_mse_az_revcorr = RNN_dict_preds_revcorr['mean_mse_az_corrected_RNN'][x,]
    radii_mse_1 = RNN_mean_mse_az
    radii_mse_2 = np.zeros(1)
    radii_mse_2[0] = RNN_mean_mse_az[0]
    radii_mse = np.concatenate((radii_mse_1, radii_mse_2)) # you have to add the last point at the end of the array to close the circle
    radii_msecor_1 = RNN_mean_mse_az_revcorr
    radii_msecor_2 = np.zeros(1)
    radii_msecor_2[0] = RNN_mean_mse_az_revcorr[0]
    radii_msecor = np.concatenate((radii_msecor_1, radii_msecor_2)) # you have to add the last point at the end of the array to close the circle
    plt.figure(figsize = (3,3))
    ax = plt.subplot(111, projection='polar')
    theta_1 = azimuthrange
    theta_2 = np.zeros(1)
    theta_2[0] = azimuthrange[0]
    theta = np.concatenate((theta_1, theta_2))
    ax.plot(np.radians(theta),radii_mse, color = 'black', linewidth = 2)
    ax.plot(np.radians(theta),radii_msecor, linestyle = '--', color = 'red', linewidth = 1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if 'MSE' in tempname:
        ax.set_ylim(0,.30)
        ax.set_yticks(np.arange(0,.30,.1))
    elif 'AD' in tempname:
        ax.set_ylim(0,.35)
        ax.set_yticks(np.arange(0,.30,.1))
    #ax.legend(('Raw', 'Reversal Corrected'), fontsize = 10,bbox_to_anchor=(1,.1)) # this latter thing is to put the legend in the correct location
    plt.grid(linestyle = ':', linewidth = 1)
    plt.savefig(dirfiles+'/PolarPlot_MSE_'+tempname+'.eps')
    plt.close('all')
# for human MSE
radii_human_mse = [.08, .09, .1, .12, .14, .16, .17, .175, .18, .18, .18, .175, .175, .172, .172, .168, .168, .168, .168, .168, .168, .172, .172, .175,.175,.180,.180,.18,.175,.17,.17, .16,.14,.12, .1,.09,.08]    
radii_human_mse_cor = [.04, .05, .07, .1, .11, .12, .14, .16, .18, .18, .18, .175, .175, .172, .172, .168, .168, .168, .168, .168, .168, .172, .172, .175,.175,.180,.180,.18,.18,.16,.14, .12,.11,.1, .07,.05,.04]    
plt.figure(figsize = (3,3))
ax = plt.subplot(111, projection='polar')
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1, theta_2))
ax.plot(np.radians(theta),radii_human_mse, color = 'black', linewidth = 2)
ax.plot(np.radians(theta),radii_human_mse_cor, linestyle = '--', color = 'red', linewidth = 1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,.30)
ax.set_yticks(np.arange(0,.30,.1))
#ax.legend(('Raw', 'Reversal Corrected'), fontsize = 10,bbox_to_anchor=(1,.1)) # this latter thing is to put the legend in the correct location
plt.grid(linestyle = ':', linewidth = 1)
plt.savefig(dirfiles+'/PolarPlot_Human_MSE_'+tempname+'.eps')
plt.close('all')

# for AD
for x in range(len(filenames_CNN)):
    tempname = filenames_CNN[x]
    tempname = tempname[:-16]
    CNN_mean_ad_az = CNN_dict_preds['mean_cosdistdeg_az'][x,]
    CNN_mean_ad_az_revcorr = CNN_dict_preds_revcorr['mean_cosdistdeg_az_corrected'][x,]
    radii_ad_1 = CNN_mean_ad_az
    radii_ad_2 = np.zeros(1)
    radii_ad_2[0] = CNN_mean_ad_az[0]
    radii_ad = np.concatenate((radii_ad_1, radii_ad_2)) # you have to add the last point at the end of the array to close the circle
    radii_adcor_1 = CNN_mean_ad_az_revcorr
    radii_adcor_2 = np.zeros(1)
    radii_adcor_2[0] = CNN_mean_ad_az_revcorr[0]
    radii_adcor = np.concatenate((radii_adcor_1, radii_adcor_2)) # you have to add the last point at the end of the array to close the circle
    plt.figure(figsize = (3,3))
    ax = plt.subplot(111, projection='polar')
    theta_1 = azimuthrange
    theta_2 = np.zeros(1)
    theta_2[0] = azimuthrange[0]
    theta = np.concatenate((theta_1, theta_2))
    ax.plot(np.radians(theta),radii_ad, color = 'black', linewidth = 2)
    ax.plot(np.radians(theta),radii_adcor, linestyle = '--', color = 'red', linewidth = 1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0,12)
    ax.set_yticks(np.arange(0,12,4))
    plt.grid(linestyle = ':', linewidth = 1)
    plt.savefig(dirfiles+'/PolarPlot_AD_'+tempname+'.eps')
    plt.close('all')
for x in range(len(filenames_RNN)):
    tempname = filenames_RNN[x]
    tempname = tempname[2:-16]
    RNN_mean_ad_az = RNN_dict_preds['mean_cosdistdeg_az_RNN'][x,]
    RNN_mean_ad_az_revcorr = RNN_dict_preds_revcorr['mean_cosdistdeg_az_corrected_RNN'][x,]
    radii_ad_1 = RNN_mean_ad_az
    radii_ad_2 = np.zeros(1)
    radii_ad_2[0] = RNN_mean_ad_az[0]
    radii_ad = np.concatenate((radii_ad_1, radii_ad_2)) # you have to add the last point at the end of the array to close the circle
    radii_adcor_1 = RNN_mean_ad_az_revcorr
    radii_adcor_2 = np.zeros(1)
    radii_adcor_2[0] = RNN_mean_ad_az_revcorr[0]
    radii_adcor = np.concatenate((radii_adcor_1, radii_adcor_2)) # you have to add the last point at the end of the array to close the circle
    plt.figure(figsize = (3,3))
    ax = plt.subplot(111, projection='polar')
    theta_1 = azimuthrange
    theta_2 = np.zeros(1)
    theta_2[0] = azimuthrange[0]
    theta = np.concatenate((theta_1, theta_2))
    ax.plot(np.radians(theta),radii_ad, color = 'black', linewidth = 2)
    ax.plot(np.radians(theta),radii_adcor, linestyle = '--', color = 'red', linewidth = 1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0,12)
    ax.set_yticks(np.arange(0,12,4))
    plt.grid(linestyle = ':', linewidth = 1)
    plt.savefig(dirfiles+'/PolarPlot_AD_'+tempname+'.eps')
    plt.close('all')
# for human AD
radii_human_ad = [4, 5, 5.8, 6.5, 7.2, 7.8, 8.3, 8.4, 8.4, 8.5, 8.5, 8.35, 8.35, 8.1, 8.1, 7.9, 7.9, 7.9, 7.9, 7.9, 7.9, 8.1, 8.1, 8.35, 8.35, 8.35, 8.5, 8.5, 8.4, 8.4, 8.3, 7.8, 7.2,6.5,5.8, 5,4]    
radii_human_ad_cor = [3, 4, 4.8, 5.5, 6.5, 7.2, 8, 8.4, 8.4, 8.5, 8.5, 8.35, 8.35, 8.1, 8.1, 7.9, 7.9, 7.9, 7.9, 7.9, 7.9, 8.1, 8.1, 8.35, 8.35, 8.35, 8.5, 8.5, 8.4, 8.4, 8, 7.2,6.5,5.5,4.8, 4,3]    
plt.figure(figsize = (3,3))
ax = plt.subplot(111, projection='polar')
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1, theta_2))
ax.plot(np.radians(theta),radii_human_ad, color = 'black', linewidth = 2)
ax.plot(np.radians(theta),radii_human_ad_cor, linestyle = '--', color = 'red', linewidth = 1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,12)
ax.set_yticks(np.arange(0,12,4))
#ax.legend(('Raw', 'Reversal Corrected'), fontsize = 10,bbox_to_anchor=(1,.1)) # this latter thing is to put the legend in the correct location
plt.grid(linestyle = ':', linewidth = 1)
plt.savefig(dirfiles+'/PolarPlot_Human_AD_'+tempname+'.eps')
plt.close('all')
    
# -----------------------------------------------------------------------------
## polar plot of the error per azimuth across best models for each model class --> but these are the models that are best without correction for reversals
# MODEL CLASS 1: for CNN trained with MSE
CNN_mse_namesbestmod = ['CNN_S_L4_K-64-64-64-128_KS-17-17-17-17_MP-12-22-22-32_DO-d2-d2-d2-d2_MSE','CNN_S_L4_K-32-32-64-128_KS-15-15-15-15_MP-12-22-22-32_DO-d2-d2-d2-d2_MSE','CNN_S_L4_K-32-32-64-64_KS-19-19-19-19_MP-12-22-22-32_DO-d2-d2-d2-d2_MSE']
# initialize matrices
CNN_mse_mse_bestmoddata = np.zeros((len(CNN_mse_namesbestmod),len(azimuthrange)))
CNN_mse_mse_bestmoddata_revcorr = np.zeros((len(CNN_mse_namesbestmod),len(azimuthrange)))
CNN_mse_ad_bestmoddata = np.zeros((len(CNN_mse_namesbestmod),len(azimuthrange)))
CNN_mse_ad_bestmoddata_revcorr = np.zeros((len(CNN_mse_namesbestmod),len(azimuthrange)))
# retrieve data
for x in range(len(CNN_mse_namesbestmod)):
    for z in range(len(filenames_CNN)):
        tempnamecheck = filenames_CNN[z]
        tempnamecheck = tempnamecheck[:-16]
        if CNN_mse_namesbestmod[x] ==  tempnamecheck:
            CNN_mse_mse_bestmoddata[x] = CNN_dict_preds['mean_mse_az'][z,]
            CNN_mse_mse_bestmoddata_revcorr[x] = CNN_dict_preds_revcorr['mean_mse_az_corrected'][z,]
            CNN_mse_ad_bestmoddata[x] = CNN_dict_preds['mean_cosdistdeg_az'][z,]
            CNN_mse_ad_bestmoddata_revcorr[x] = CNN_dict_preds_revcorr['mean_cosdistdeg_az_corrected'][z,]
            print(z)
            break # stop if the right model has been found
# compute average and standard devation
CNN_mse_mse_bestmod_avg = np.mean(CNN_mse_mse_bestmoddata, axis = 0)
CNN_mse_mse_bestmod_stdev = np.std(CNN_mse_mse_bestmoddata, axis = 0)
CNN_mse_mse_revcorr_bestmod_avg = np.mean(CNN_mse_mse_bestmoddata_revcorr, axis = 0)
CNN_mse_mse_revcorr_bestmod_stdev = np.std(CNN_mse_mse_bestmoddata_revcorr, axis = 0)
CNN_mse_ad_bestmod_avg = np.mean(CNN_mse_ad_bestmoddata, axis = 0)
CNN_mse_ad_bestmod_stdev = np.std(CNN_mse_ad_bestmoddata, axis = 0)
CNN_mse_ad_revcorr_bestmod_avg = np.mean(CNN_mse_ad_bestmoddata_revcorr, axis = 0)
CNN_mse_ad_revcorr_bestmod_stdev = np.std(CNN_mse_ad_bestmoddata_revcorr, axis = 0)

# plot MSE_mse
# create range of angles
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1,theta_2))
radii_mse_mse_bestmod_1 = CNN_mse_mse_bestmod_avg
radii_mse_mse_bestmod_2 = np.zeros(1)
radii_mse_mse_bestmod_2[0] = CNN_mse_mse_bestmod_avg[0]
radii_mse_mse_bestmod = np.concatenate((radii_mse_mse_bestmod_1, radii_mse_mse_bestmod_2))
stdinterval_mse_mse_bestmod_1 = CNN_mse_mse_bestmod_stdev
stdinterval_mse_mse_bestmod_2 = np.zeros(1)
stdinterval_mse_mse_bestmod_2[0] = CNN_mse_mse_bestmod_stdev[0]
stdinterval_mse_mse_bestmod = np.concatenate((stdinterval_mse_mse_bestmod_1,stdinterval_mse_mse_bestmod_2))
radii_mse_mse_revcorr_bestmod_1 = CNN_mse_mse_revcorr_bestmod_avg
radii_mse_mse_revcorr_bestmod_2 = np.zeros(1)
radii_mse_mse_revcorr_bestmod_2[0] = CNN_mse_mse_revcorr_bestmod_avg[0]
radii_mse_mse_revcorr_bestmod = np.concatenate((radii_mse_mse_revcorr_bestmod_1,radii_mse_mse_revcorr_bestmod_2))
stdinterval_mse_mse_revcorr_bestmod_1 = CNN_mse_mse_revcorr_bestmod_stdev
stdinterval_mse_mse_revcorr_bestmod_2 = np.zeros(1)
stdinterval_mse_mse_revcorr_bestmod_2[0] = CNN_mse_mse_revcorr_bestmod_stdev[0]
stdinterval_mse_mse_revcorr_bestmod = np.concatenate((stdinterval_mse_mse_revcorr_bestmod_1,stdinterval_mse_mse_revcorr_bestmod_2))
# create standard deviation
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta),radii_mse_mse_bestmod, color = 'black', linewidth = 3)
ax.plot(np.radians(theta),radii_mse_mse_revcorr_bestmod, linestyle = '--', color = 'red', linewidth = 1.5)
ax.fill_between(np.radians(theta), (radii_mse_mse_bestmod-stdinterval_mse_mse_bestmod), (radii_mse_mse_bestmod+stdinterval_mse_mse_bestmod), color='black', alpha=.2)
ax.fill_between(np.radians(theta), (radii_mse_mse_revcorr_bestmod-stdinterval_mse_mse_revcorr_bestmod), (radii_mse_mse_revcorr_bestmod+stdinterval_mse_mse_revcorr_bestmod), color='red', alpha=.1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,.45)
ax.set_yticks(np.arange(0,.45,.1))
plt.savefig(dirfiles+'/PolarPlot_BestCNN_TrainedWithMSE_MSEscore_perAzimuth.png')


# plot MSE_ad
# create range of angles
radii_mse_ad_bestmod_1 = CNN_mse_ad_bestmod_avg
radii_mse_ad_bestmod_2 = np.zeros(1)
radii_mse_ad_bestmod_2[0] = CNN_mse_ad_bestmod_avg[0]
radii_mse_ad_bestmod = np.concatenate((radii_mse_ad_bestmod_1, radii_mse_ad_bestmod_2))
stdinterval_mse_ad_bestmod_1 = CNN_mse_ad_bestmod_stdev
stdinterval_mse_ad_bestmod_2 = np.zeros(1)
stdinterval_mse_ad_bestmod_2[0] = CNN_mse_ad_bestmod_stdev[0]
stdinterval_mse_ad_bestmod = np.concatenate((stdinterval_mse_ad_bestmod_1,stdinterval_mse_ad_bestmod_2))
radii_mse_ad_revcorr_bestmod_1 = CNN_mse_ad_revcorr_bestmod_avg
radii_mse_ad_revcorr_bestmod_2 = np.zeros(1)
radii_mse_ad_revcorr_bestmod_2[0] = CNN_mse_ad_revcorr_bestmod_avg[0]
radii_mse_ad_revcorr_bestmod = np.concatenate((radii_mse_ad_revcorr_bestmod_1,radii_mse_ad_revcorr_bestmod_2))
stdinterval_mse_ad_revcorr_bestmod_1 = CNN_mse_ad_revcorr_bestmod_stdev
stdinterval_mse_ad_revcorr_bestmod_2 = np.zeros(1)
stdinterval_mse_ad_revcorr_bestmod_2[0] = CNN_mse_ad_revcorr_bestmod_stdev[0]
stdinterval_mse_ad_revcorr_bestmod = np.concatenate((stdinterval_mse_ad_revcorr_bestmod_1,stdinterval_mse_ad_revcorr_bestmod_2))
# create standard deviation
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta),radii_mse_ad_bestmod, color = 'black', linewidth = 3)
ax.plot(np.radians(theta),radii_mse_ad_revcorr_bestmod, linestyle = '--', color = 'red', linewidth = 1.5)
ax.fill_between(np.radians(theta), (radii_mse_ad_bestmod-stdinterval_mse_ad_bestmod), (radii_mse_ad_bestmod+stdinterval_mse_ad_bestmod), color='black', alpha=.2)
ax.fill_between(np.radians(theta), (radii_mse_ad_revcorr_bestmod-stdinterval_mse_ad_revcorr_bestmod), (radii_mse_ad_revcorr_bestmod+stdinterval_mse_ad_revcorr_bestmod), color='red', alpha=.1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,20)
ax.set_yticks(np.arange(0,20,4))
plt.savefig(dirfiles+'/PolarPlot_BestCNN_TrainedWithMSE_ADscore_perAzimuth.png')

 
## MODEL CLASS 2: for CNN model trained with AD
CNN_ad_namesbestmod = ['CNN_S_L4_K-64-64-64-128_KS-15-15-15-15_MP-12-22-22-32_DO-d2-d2-d2-d2_AD','CNN_S_L4_K-32-32-64-128_KS-17-17-17-17_MP-12-22-22-32_DO-d2-d2-d2-d2_AD','CNN_S_L4_K-32-32-64-64_KS-17-17-17-17_MP-12-22-22-32_DO-d2-d2-d2-d2_AD']
# intialize matrices
CNN_ad_ad_bestmoddata = np.zeros((len(CNN_ad_namesbestmod),len(azimuthrange)))
CNN_ad_ad_bestmoddata_revcorr = np.zeros((len(CNN_ad_namesbestmod),len(azimuthrange)))
CNN_ad_mse_bestmoddata = np.zeros((len(CNN_ad_namesbestmod),len(azimuthrange)))
CNN_ad_mse_bestmoddata_revcorr = np.zeros((len(CNN_ad_namesbestmod),len(azimuthrange)))

# retrieve data
for x in range(len(CNN_ad_namesbestmod)):
    for z in range(len(filenames_CNN)):
        tempnamecheck = filenames_CNN[z]
        tempnamecheck = tempnamecheck[:-16]
        if CNN_ad_namesbestmod[x] ==  tempnamecheck:
            CNN_ad_ad_bestmoddata[x] = CNN_dict_preds['mean_cosdistdeg_az'][z,]
            CNN_ad_ad_bestmoddata_revcorr[x] = CNN_dict_preds_revcorr['mean_cosdistdeg_az_corrected'][z,]
            CNN_ad_mse_bestmoddata[x] = CNN_dict_preds['mean_mse_az'][z,]
            CNN_ad_mse_bestmoddata_revcorr[x] = CNN_dict_preds_revcorr['mean_mse_az_corrected'][z,]
            print(z)
            break # stop if the right model has been found
            
# compute average and standard deviation
CNN_ad_ad_bestmod_avg = np.mean(CNN_ad_ad_bestmoddata, axis = 0)
CNN_ad_ad_bestmod_stdev = np.std(CNN_ad_ad_bestmoddata, axis = 0)
CNN_ad_ad_revcorr_bestmod_avg = np.mean(CNN_ad_ad_bestmoddata_revcorr, axis = 0)
CNN_ad_ad_revcorr_bestmod_stdev = np.std(CNN_ad_ad_bestmoddata_revcorr, axis = 0)
CNN_ad_mse_bestmod_avg = np.mean(CNN_ad_mse_bestmoddata, axis = 0)
CNN_ad_mse_bestmod_stdev = np.std(CNN_ad_mse_bestmoddata, axis = 0)
CNN_ad_mse_revcorr_bestmod_avg = np.mean(CNN_ad_mse_bestmoddata_revcorr, axis = 0)
CNN_ad_mse_revcorr_bestmod_stdev = np.std(CNN_ad_mse_bestmoddata_revcorr, axis = 0)  


# plot AD_mse
# create range of angles
radii_ad_mse_bestmod_1 = CNN_ad_mse_bestmod_avg
radii_ad_mse_bestmod_2 = np.zeros(1)
radii_ad_mse_bestmod_2[0] = CNN_ad_mse_bestmod_avg[0]
radii_ad_mse_bestmod = np.concatenate((radii_ad_mse_bestmod_1, radii_ad_mse_bestmod_2))
stdinterval_ad_mse_bestmod_1 = CNN_ad_mse_bestmod_stdev
stdinterval_ad_mse_bestmod_2 = np.zeros(1)
stdinterval_ad_mse_bestmod_2[0] = CNN_ad_mse_bestmod_stdev[0]
stdinterval_ad_mse_bestmod = np.concatenate((stdinterval_ad_mse_bestmod_1,stdinterval_ad_mse_bestmod_2))
radii_ad_mse_revcorr_bestmod_1 = CNN_ad_mse_revcorr_bestmod_avg
radii_ad_mse_revcorr_bestmod_2 = np.zeros(1)
radii_ad_mse_revcorr_bestmod_2[0] = CNN_ad_mse_revcorr_bestmod_avg[0]
radii_ad_mse_revcorr_bestmod = np.concatenate((radii_ad_mse_revcorr_bestmod_1,radii_ad_mse_revcorr_bestmod_2))
stdinterval_ad_mse_revcorr_bestmod_1 = CNN_ad_mse_revcorr_bestmod_stdev
stdinterval_ad_mse_revcorr_bestmod_2 = np.zeros(1)
stdinterval_ad_mse_revcorr_bestmod_2[0] = CNN_ad_mse_revcorr_bestmod_stdev[0]
stdinterval_ad_mse_revcorr_bestmod = np.concatenate((stdinterval_ad_mse_revcorr_bestmod_1,stdinterval_ad_mse_revcorr_bestmod_2))
# create standard deviation
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta),radii_ad_mse_bestmod, color = 'black', linewidth = 3)
ax.plot(np.radians(theta),radii_ad_mse_revcorr_bestmod, linestyle = '--', color = 'red', linewidth = 1.5)
ax.fill_between(np.radians(theta), (radii_ad_mse_bestmod-stdinterval_ad_mse_bestmod), (radii_ad_mse_bestmod+stdinterval_ad_mse_bestmod), color='black', alpha=.2)
ax.fill_between(np.radians(theta), (radii_ad_mse_revcorr_bestmod-stdinterval_ad_mse_revcorr_bestmod), (radii_ad_mse_revcorr_bestmod+stdinterval_ad_mse_revcorr_bestmod), color='red', alpha=.1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,.45)
ax.set_yticks(np.arange(0,.45,.1))
plt.savefig(dirfiles+'/PolarPlot_BestCNN_TrainedWithAD_MSEscore_perAzimuth.png')


# plot AD_ad
# create range of angles
radii_ad_ad_bestmod_1 = CNN_ad_ad_bestmod_avg
radii_ad_ad_bestmod_2 = np.zeros(1)
radii_ad_ad_bestmod_2[0] = CNN_ad_ad_bestmod_avg[0]
radii_ad_ad_bestmod = np.concatenate((radii_ad_ad_bestmod_1, radii_ad_ad_bestmod_2))
stdinterval_ad_ad_bestmod_1 = CNN_ad_ad_bestmod_stdev
stdinterval_ad_ad_bestmod_2 = np.zeros(1)
stdinterval_ad_ad_bestmod_2[0] = CNN_ad_ad_bestmod_stdev[0]
stdinterval_ad_ad_bestmod = np.concatenate((stdinterval_ad_ad_bestmod_1,stdinterval_ad_ad_bestmod_2))
radii_ad_ad_revcorr_bestmod_1 = CNN_ad_ad_revcorr_bestmod_avg
radii_ad_ad_revcorr_bestmod_2 = np.zeros(1)
radii_ad_ad_revcorr_bestmod_2[0] = CNN_ad_ad_revcorr_bestmod_avg[0]
radii_ad_ad_revcorr_bestmod = np.concatenate((radii_ad_ad_revcorr_bestmod_1,radii_ad_ad_revcorr_bestmod_2))
stdinterval_ad_ad_revcorr_bestmod_1 = CNN_ad_ad_revcorr_bestmod_stdev
stdinterval_ad_ad_revcorr_bestmod_2 = np.zeros(1)
stdinterval_ad_ad_revcorr_bestmod_2[0] = CNN_ad_ad_revcorr_bestmod_stdev[0]
stdinterval_ad_ad_revcorr_bestmod = np.concatenate((stdinterval_ad_ad_revcorr_bestmod_1,stdinterval_ad_ad_revcorr_bestmod_2))
# create standard deviation
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta),radii_ad_ad_bestmod, color = 'black', linewidth = 3)
ax.plot(np.radians(theta),radii_ad_ad_revcorr_bestmod, linestyle = '--', color = 'red', linewidth = 1.5)
ax.fill_between(np.radians(theta), (radii_ad_ad_bestmod-stdinterval_ad_ad_bestmod), (radii_ad_ad_bestmod+stdinterval_mse_ad_bestmod), color='black', alpha=.2)
ax.fill_between(np.radians(theta), (radii_ad_ad_revcorr_bestmod-stdinterval_ad_ad_revcorr_bestmod), (radii_ad_ad_revcorr_bestmod+stdinterval_ad_ad_revcorr_bestmod), color='red', alpha=.1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,20)
ax.set_yticks(np.arange(0,20,4))
plt.savefig(dirfiles+'/PolarPlot_BestCNN_TrainedWithAD_ADscore_perAzimuth.png')


# MODEL CLASS 3: RNN trained with MSE
# initialize matrices
RNN_mse_mse_data = np.zeros((int(len(filenames_RNN)/2),len(azimuthrange))) # half of the models are trained with MSE, the other half with AD
RNN_mse_mse_data_revcorr = np.zeros((int(len(filenames_RNN)/2),len(azimuthrange)))
RNN_mse_ad_data = np.zeros((int(len(filenames_RNN)/2),len(azimuthrange)))
RNN_mse_ad_data_revcorr = np.zeros((int(len(filenames_RNN)/2),len(azimuthrange)))
RNN_ad_mse_data = np.zeros((int(len(filenames_RNN)/2),len(azimuthrange))) # half of the models are trained with MSE, the other half with AD
RNN_ad_mse_data_revcorr = np.zeros((int(len(filenames_RNN)/2),len(azimuthrange)))
RNN_ad_ad_data = np.zeros((int(len(filenames_RNN)/2),len(azimuthrange)))
RNN_ad_ad_data_revcorr = np.zeros((int(len(filenames_RNN)/2),len(azimuthrange)))
cnt_mse = 0 # initialize counters
cnt_ad = 0
# retrieve data
for x in range(len(filenames_RNN)):
   tempname = filenames_RNN[x]
   if 'MSE' in tempname:
       RNN_mse_mse_data[cnt_mse] = RNN_dict_preds['mean_mse_az_RNN'][x]
       RNN_mse_mse_data_revcorr[cnt_mse] = RNN_dict_preds_revcorr['mean_mse_az_corrected_RNN'][x]
       RNN_mse_ad_data[cnt_mse] = RNN_dict_preds['mean_cosdistdeg_az_RNN'][x]
       RNN_mse_ad_data_revcorr[cnt_mse] = RNN_dict_preds_revcorr['mean_cosdistdeg_az_corrected_RNN'][x]
       cnt_mse = cnt_mse +1
   elif 'AD' in tempname:
       RNN_ad_mse_data[cnt_ad] = RNN_dict_preds['mean_mse_az_RNN'][x]
       RNN_ad_mse_data_revcorr[cnt_ad] = RNN_dict_preds_revcorr['mean_mse_az_corrected_RNN'][x]
       RNN_ad_ad_data[cnt_ad] = RNN_dict_preds['mean_cosdistdeg_az_RNN'][x]
       RNN_ad_ad_data_revcorr[cnt_ad] = RNN_dict_preds_revcorr['mean_cosdistdeg_az_corrected_RNN'][x]
       cnt_ad = cnt_ad+1
       
# compute average and standard devation
RNN_mse_mse_bestmod_avg = np.mean(RNN_mse_mse_data, axis = 0)
RNN_mse_mse_bestmod_stdev = np.std(RNN_mse_mse_data, axis = 0)
RNN_mse_mse_revcorr_bestmod_avg = np.mean(RNN_mse_mse_data_revcorr, axis = 0)
RNN_mse_mse_revcorr_bestmod_stdev = np.std(RNN_mse_mse_data_revcorr, axis = 0)
RNN_mse_ad_bestmod_avg = np.mean(RNN_mse_ad_data, axis = 0)
RNN_mse_ad_bestmod_stdev = np.std(RNN_mse_ad_data, axis = 0)
RNN_mse_ad_revcorr_bestmod_avg = np.mean(RNN_mse_ad_data_revcorr, axis = 0)
RNN_mse_ad_revcorr_bestmod_stdev = np.std(RNN_mse_ad_data_revcorr, axis = 0)

# plot MSE_mse
# create range of angles
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1,theta_2))
radii_mse_mse_bestmod_1 = RNN_mse_mse_bestmod_avg
radii_mse_mse_bestmod_2 = np.zeros(1)
radii_mse_mse_bestmod_2[0] = RNN_mse_mse_bestmod_avg[0]
radii_mse_mse_bestmod = np.concatenate((radii_mse_mse_bestmod_1, radii_mse_mse_bestmod_2))
stdinterval_mse_mse_bestmod_1 = RNN_mse_mse_bestmod_stdev
stdinterval_mse_mse_bestmod_2 = np.zeros(1)
stdinterval_mse_mse_bestmod_2[0] = RNN_mse_mse_bestmod_stdev[0]
stdinterval_mse_mse_bestmod = np.concatenate((stdinterval_mse_mse_bestmod_1,stdinterval_mse_mse_bestmod_2))
radii_mse_mse_revcorr_bestmod_1 = RNN_mse_mse_revcorr_bestmod_avg
radii_mse_mse_revcorr_bestmod_2 = np.zeros(1)
radii_mse_mse_revcorr_bestmod_2[0] = RNN_mse_mse_revcorr_bestmod_avg[0]
radii_mse_mse_revcorr_bestmod = np.concatenate((radii_mse_mse_revcorr_bestmod_1,radii_mse_mse_revcorr_bestmod_2))
stdinterval_mse_mse_revcorr_bestmod_1 = RNN_mse_mse_revcorr_bestmod_stdev
stdinterval_mse_mse_revcorr_bestmod_2 = np.zeros(1)
stdinterval_mse_mse_revcorr_bestmod_2[0] = RNN_mse_mse_revcorr_bestmod_stdev[0]
stdinterval_mse_mse_revcorr_bestmod = np.concatenate((stdinterval_mse_mse_revcorr_bestmod_1,stdinterval_mse_mse_revcorr_bestmod_2))
# create standard deviation
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta),radii_mse_mse_bestmod, color = 'black', linewidth = 3)
ax.plot(np.radians(theta),radii_mse_mse_revcorr_bestmod, linestyle = '--', color = 'red', linewidth = 1.5)
ax.fill_between(np.radians(theta), (radii_mse_mse_bestmod-stdinterval_mse_mse_bestmod), (radii_mse_mse_bestmod+stdinterval_mse_mse_bestmod), color='black', alpha=.2)
ax.fill_between(np.radians(theta), (radii_mse_mse_revcorr_bestmod-stdinterval_mse_mse_revcorr_bestmod), (radii_mse_mse_revcorr_bestmod+stdinterval_mse_mse_revcorr_bestmod), color='red', alpha=.1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,.45)
ax.set_yticks(np.arange(0,.45,.1))
plt.savefig(dirfiles+'/PolarPlot_BestRNN_TrainedWithMSE_MSEscore_perAzimuth.png')

# plot MSE_ad
# create range of angles
radii_mse_ad_bestmod_1 = RNN_mse_ad_bestmod_avg
radii_mse_ad_bestmod_2 = np.zeros(1)
radii_mse_ad_bestmod_2[0] = RNN_mse_ad_bestmod_avg[0]
radii_mse_ad_bestmod = np.concatenate((radii_mse_ad_bestmod_1, radii_mse_ad_bestmod_2))
stdinterval_mse_ad_bestmod_1 = RNN_mse_ad_bestmod_stdev
stdinterval_mse_ad_bestmod_2 = np.zeros(1)
stdinterval_mse_ad_bestmod_2[0] = RNN_mse_ad_bestmod_stdev[0]
stdinterval_mse_ad_bestmod = np.concatenate((stdinterval_mse_ad_bestmod_1,stdinterval_mse_ad_bestmod_2))
radii_mse_ad_revcorr_bestmod_1 = RNN_mse_ad_revcorr_bestmod_avg
radii_mse_ad_revcorr_bestmod_2 = np.zeros(1)
radii_mse_ad_revcorr_bestmod_2[0] = RNN_mse_ad_revcorr_bestmod_avg[0]
radii_mse_ad_revcorr_bestmod = np.concatenate((radii_mse_ad_revcorr_bestmod_1,radii_mse_ad_revcorr_bestmod_2))
stdinterval_mse_ad_revcorr_bestmod_1 = RNN_mse_ad_revcorr_bestmod_stdev
stdinterval_mse_ad_revcorr_bestmod_2 = np.zeros(1)
stdinterval_mse_ad_revcorr_bestmod_2[0] = RNN_mse_ad_revcorr_bestmod_stdev[0]
stdinterval_mse_ad_revcorr_bestmod = np.concatenate((stdinterval_mse_ad_revcorr_bestmod_1,stdinterval_mse_ad_revcorr_bestmod_2))
# create standard deviation
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta),radii_mse_ad_bestmod, color = 'black', linewidth = 3)
ax.plot(np.radians(theta),radii_mse_ad_revcorr_bestmod, linestyle = '--', color = 'red', linewidth = 1.5)
ax.fill_between(np.radians(theta), (radii_mse_ad_bestmod-stdinterval_mse_ad_bestmod), (radii_mse_ad_bestmod+stdinterval_mse_ad_bestmod), color='black', alpha=.2)
ax.fill_between(np.radians(theta), (radii_mse_ad_revcorr_bestmod-stdinterval_mse_ad_revcorr_bestmod), (radii_mse_ad_revcorr_bestmod+stdinterval_mse_ad_revcorr_bestmod), color='red', alpha=.1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,20)
ax.set_yticks(np.arange(0,20,4))
plt.savefig(dirfiles+'/PolarPlot_BestRNN_TrainedWithMSE_ADscore_perAzimuth.png')

### MODEL CLASS 4: For RNN trained with AD

# compute average and standard devation
RNN_ad_mse_bestmod_avg = np.mean(RNN_ad_mse_data, axis = 0)
RNN_ad_mse_bestmod_stdev = np.std(RNN_ad_mse_data, axis = 0)
RNN_ad_mse_revcorr_bestmod_avg = np.mean(RNN_ad_mse_data_revcorr, axis = 0)
RNN_ad_mse_revcorr_bestmod_stdev = np.std(RNN_ad_mse_data_revcorr, axis = 0)
RNN_ad_ad_bestmod_avg = np.mean(RNN_ad_ad_data, axis = 0)
RNN_ad_ad_bestmod_stdev = np.std(RNN_ad_ad_data, axis = 0)
RNN_ad_ad_revcorr_bestmod_avg = np.mean(RNN_ad_ad_data_revcorr, axis = 0)
RNN_ad_ad_revcorr_bestmod_stdev = np.std(RNN_ad_ad_data_revcorr, axis = 0)

# plot AD_mse
# create range of angles
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1,theta_2))
radii_ad_mse_bestmod_1 = RNN_ad_mse_bestmod_avg
radii_ad_mse_bestmod_2 = np.zeros(1)
radii_ad_mse_bestmod_2[0] = RNN_ad_mse_bestmod_avg[0]
radii_ad_mse_bestmod = np.concatenate((radii_ad_mse_bestmod_1, radii_ad_mse_bestmod_2))
stdinterval_ad_mse_bestmod_1 = RNN_ad_mse_bestmod_stdev
stdinterval_ad_mse_bestmod_2 = np.zeros(1)
stdinterval_ad_mse_bestmod_2[0] = RNN_ad_mse_bestmod_stdev[0]
stdinterval_ad_mse_bestmod = np.concatenate((stdinterval_ad_mse_bestmod_1,stdinterval_ad_mse_bestmod_2))
radii_ad_mse_revcorr_bestmod_1 = RNN_ad_mse_revcorr_bestmod_avg
radii_ad_mse_revcorr_bestmod_2 = np.zeros(1)
radii_ad_mse_revcorr_bestmod_2[0] = RNN_ad_mse_revcorr_bestmod_avg[0]
radii_ad_mse_revcorr_bestmod = np.concatenate((radii_ad_mse_revcorr_bestmod_1,radii_ad_mse_revcorr_bestmod_2))
stdinterval_ad_mse_revcorr_bestmod_1 = RNN_ad_mse_revcorr_bestmod_stdev
stdinterval_ad_mse_revcorr_bestmod_2 = np.zeros(1)
stdinterval_ad_mse_revcorr_bestmod_2[0] = RNN_ad_mse_revcorr_bestmod_stdev[0]
stdinterval_ad_mse_revcorr_bestmod = np.concatenate((stdinterval_ad_mse_revcorr_bestmod_1,stdinterval_ad_mse_revcorr_bestmod_2))
# create standard deviation
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta),radii_ad_mse_bestmod, color = 'black', linewidth = 3)
ax.plot(np.radians(theta),radii_ad_mse_revcorr_bestmod, linestyle = '--', color = 'red', linewidth = 1.5)
ax.fill_between(np.radians(theta), (radii_ad_mse_bestmod-stdinterval_ad_mse_bestmod), (radii_mse_mse_bestmod+stdinterval_mse_mse_bestmod), color='black', alpha=.2)
ax.fill_between(np.radians(theta), (radii_ad_mse_revcorr_bestmod-stdinterval_ad_mse_revcorr_bestmod), (radii_mse_mse_revcorr_bestmod+stdinterval_mse_mse_revcorr_bestmod), color='red', alpha=.1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,.45)
ax.set_yticks(np.arange(0,.45,.1))
plt.savefig(dirfiles+'/PolarPlot_BestRNN_TrainedWithAD_MSEscore_perAzimuth.png')

# plot AD_ad
# create range of angles
radii_mse_ad_bestmod_1 = RNN_mse_ad_bestmod_avg
radii_mse_ad_bestmod_2 = np.zeros(1)
radii_mse_ad_bestmod_2[0] = RNN_mse_ad_bestmod_avg[0]
radii_mse_ad_bestmod = np.concatenate((radii_mse_ad_bestmod_1, radii_mse_ad_bestmod_2))
stdinterval_mse_ad_bestmod_1 = RNN_mse_ad_bestmod_stdev
stdinterval_mse_ad_bestmod_2 = np.zeros(1)
stdinterval_mse_ad_bestmod_2[0] = RNN_mse_ad_bestmod_stdev[0]
stdinterval_mse_ad_bestmod = np.concatenate((stdinterval_mse_ad_bestmod_1,stdinterval_mse_ad_bestmod_2))
radii_mse_ad_revcorr_bestmod_1 = RNN_mse_ad_revcorr_bestmod_avg
radii_mse_ad_revcorr_bestmod_2 = np.zeros(1)
radii_mse_ad_revcorr_bestmod_2[0] = RNN_mse_ad_revcorr_bestmod_avg[0]
radii_mse_ad_revcorr_bestmod = np.concatenate((radii_mse_ad_revcorr_bestmod_1,radii_mse_ad_revcorr_bestmod_2))
stdinterval_mse_ad_revcorr_bestmod_1 = RNN_mse_ad_revcorr_bestmod_stdev
stdinterval_mse_ad_revcorr_bestmod_2 = np.zeros(1)
stdinterval_mse_ad_revcorr_bestmod_2[0] = RNN_mse_ad_revcorr_bestmod_stdev[0]
stdinterval_mse_ad_revcorr_bestmod = np.concatenate((stdinterval_mse_ad_revcorr_bestmod_1,stdinterval_mse_ad_revcorr_bestmod_2))
# create standard deviation
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta),radii_mse_ad_bestmod, color = 'black', linewidth = 3)
ax.plot(np.radians(theta),radii_mse_ad_revcorr_bestmod, linestyle = '--', color = 'red', linewidth = 1.5)
ax.fill_between(np.radians(theta), (radii_mse_ad_bestmod-stdinterval_mse_ad_bestmod), (radii_mse_ad_bestmod+stdinterval_mse_ad_bestmod), color='black', alpha=.2)
ax.fill_between(np.radians(theta), (radii_mse_ad_revcorr_bestmod-stdinterval_mse_ad_revcorr_bestmod), (radii_mse_ad_revcorr_bestmod+stdinterval_mse_ad_revcorr_bestmod), color='red', alpha=.1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0,20)
ax.set_yticks(np.arange(0,20,4))
plt.savefig(dirfiles+'/PolarPlot_BestRNN_TrainedWithAD_ADscore_perAzimuth.png')


#-------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------


## Plot of the effect of kernel size on percentage of front-back reversals
# prepare data
# first model type trained with MSE
CNN_S_L4_K323264128_mse_1 = CNN_dict_preds_revcorr['reversals_percentage'][[3,5,7,9]] # get values KS 1x3 until 1x9
CNN_S_L4_K323264128_mse_1 = np.reshape(CNN_S_L4_K323264128_mse_1,len(CNN_S_L4_K323264128_mse_1,)) 
CNN_S_L4_K323264128_mse_2 = CNN_dict_preds_revcorr['reversals_percentage'][1] # get value KS 1x11
CNN_S_L4_K323264128_mse_2 = np.reshape(CNN_S_L4_K323264128_mse_2,1)
CNN_S_L4_K323264128_mse = np.concatenate((CNN_S_L4_K323264128_mse_1,CNN_S_L4_K323264128_mse_2))
# second model type trained with MSE
CNN_S_L4_K32326464_mse_1 = CNN_dict_preds_revcorr['reversals_percentage'][[13,15,17,19]] # get values KS 1x3 until 1x9
CNN_S_L4_K32326464_mse_1 = np.reshape(CNN_S_L4_K32326464_mse_1,len(CNN_S_L4_K32326464_mse_1,))
CNN_S_L4_K32326464_mse_2 = CNN_dict_preds_revcorr['reversals_percentage'][11] # get values KS 1x3 until 1x9
CNN_S_L4_K32326464_mse_2 = np.reshape(CNN_S_L4_K32326464_mse_2,1)
CNN_S_L4_K32326464_mse = np.concatenate((CNN_S_L4_K32326464_mse_1,CNN_S_L4_K32326464_mse_2))
# third model type trained with MSE
CNN_S_L4_K646464128_mse_1 = CNN_dict_preds_revcorr['reversals_percentage'][[23,25,27,29]]  # get values KS 1x3 until 1x9
CNN_S_L4_K646464128_mse_1 = np.reshape(CNN_S_L4_K646464128_mse_1,len(CNN_S_L4_K646464128_mse_1,))
CNN_S_L4_K646464128_mse_2 = CNN_dict_preds_revcorr['reversals_percentage'][21]  # get values KS 1x3 until 1x9
CNN_S_L4_K646464128_mse_2 = np.reshape(CNN_S_L4_K646464128_mse_2,1)
CNN_S_L4_K646464128_mse = np.concatenate((CNN_S_L4_K646464128_mse_1,CNN_S_L4_K646464128_mse_2))
# first model type trained with AD
CNN_S_L4_K323264128_ad_1 = CNN_dict_preds_revcorr['reversals_percentage'][[2,4,6,8]]# get values KS 1x3 until 1x9
CNN_S_L4_K323264128_ad_1 = np.reshape(CNN_S_L4_K323264128_ad_1, len(CNN_S_L4_K323264128_ad_1,))
CNN_S_L4_K323264128_ad_2 = CNN_dict_preds_revcorr['reversals_percentage'][0]# get values KS 1x3 until 1x9
CNN_S_L4_K323264128_ad_2 = np.reshape(CNN_S_L4_K323264128_ad_2, 1)
CNN_S_L4_K323264128_ad = np.concatenate((CNN_S_L4_K323264128_ad_1,CNN_S_L4_K323264128_ad_2))
# second model type trained with AD
CNN_S_L4_K32326464_ad_1 = CNN_dict_preds_revcorr['reversals_percentage'][[12,14,16,18]]# get values KS 1x3 until 1x9
CNN_S_L4_K32326464_ad_1 = np.reshape(CNN_S_L4_K32326464_ad_1,len(CNN_S_L4_K32326464_ad_1,))
CNN_S_L4_K32326464_ad_2 = CNN_dict_preds_revcorr['reversals_percentage'][10]# get values KS 1x3 until 1x9
CNN_S_L4_K32326464_ad_2 = np.reshape(CNN_S_L4_K32326464_ad_2,1)
CNN_S_L4_K32326464_ad = np.concatenate((CNN_S_L4_K32326464_ad_1,CNN_S_L4_K32326464_ad_2))
# third model type trained with AD
CNN_S_L4_K646464128_ad_1 = CNN_dict_preds_revcorr['reversals_percentage'][[22,24,26,28]]  # get values KS 1x3 until 1x9
CNN_S_L4_K646464128_ad_1 = np.reshape(CNN_S_L4_K646464128_ad_1,len(CNN_S_L4_K646464128_ad_1,))
CNN_S_L4_K646464128_ad_2 = CNN_dict_preds_revcorr['reversals_percentage'][20]  # get values KS 1x3 until 1x9
CNN_S_L4_K646464128_ad_2 = np.reshape(CNN_S_L4_K646464128_ad_2,1)
CNN_S_L4_K646464128_ad = np.concatenate((CNN_S_L4_K646464128_ad_1,CNN_S_L4_K646464128_ad_2))


# plot the data
plt.figure()
plt.plot(CNN_S_L4_K32326464_mse, linestyle = ':', marker = 'o', color = 'black')
plt.plot(CNN_S_L4_K323264128_mse, linestyle = '--', marker = 'o', color = 'black')
plt.plot(CNN_S_L4_K646464128_mse, marker = 'o', color = 'black')
plotavg_mse = np.mean(np.concatenate(([CNN_S_L4_K323264128_mse],[CNN_S_L4_K32326464_mse],[CNN_S_L4_K646464128_mse])),axis = 0)
plt.plot(plotavg_mse, marker = 'o', markersize = 7, color = 'red', linewidth = 3)
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
plt.ylim(0,.025)
plt.xlabel('Kernel size', fontsize = 12, fontweight = 'bold')
plt.xticks([0,1,2,3,4],['1x3','1x5','1x7','1x9','1x11'])
plt.ylabel('Front-back reversals (%)', fontsize = 12, fontweight = 'bold')
plt.legend(['Kernels per layer: 32-32-64-64', 'Kernels per layer: 32-32-64-128', 'Kernels per layer: 64-64-64-128', 'Average across model types'], fontsize = 10, loc = 'upper right')
plt.savefig(dirfiles+'/LinePlot_EffectKernelSize_onFrontBackReversals_CNNmodels_withMSE_Python.eps')
plt.figure()

plt.figure()
plt.plot(CNN_S_L4_K32326464_ad, linestyle = ':', marker = 'o', color = 'black')
plt.plot(CNN_S_L4_K323264128_ad, linestyle = '--', marker = 'o', color = 'black')
plt.plot(CNN_S_L4_K646464128_ad, marker = 'o', color = 'black')
plotavg_ad = np.mean(np.concatenate(([CNN_S_L4_K323264128_ad],[CNN_S_L4_K32326464_ad],[CNN_S_L4_K646464128_ad])),axis = 0)
plt.plot(plotavg_ad, marker = 'o', markersize = 7, color = 'red', linewidth = 3)
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
plt.ylim(0,.05)
plt.xlabel('Kernel size', fontsize = 12, fontweight = 'bold')
plt.xticks([0,1,2,3,4],['1x3','1x5','1x7','1x9','1x11'])
plt.ylabel('Front-back reversals (%)', fontsize = 12, fontweight = 'bold')
plt.ylim(0,0.025)
plt.legend(['Kernels per layer: 32-32-64-64', 'Kernels per layer: 32-32-64-128', 'Kernels per layer: 64-64-64-128', 'Average across model types'], fontsize = 10, loc = 'upper right')
plt.savefig(dirfiles+'/LinePlot_EffectKernelSize_onFrontBackReversals_CNNmodels_withAD_Python.eps')

leftaxis = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
rightaxis = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
azimuthrange_rotated = np.concatenate((np.take(azimuthrange,leftaxis),np.take(azimuthrange,rightaxis)))
mean_cosdistdeg_az_RNN
mean_cosdistdeg_az    

# Compute pie slices
N = 36
theta = azimuthrange
radii = mean_cosdistdeg_az_RNN[1,]
width = 1
colors = plt.cm.viridis(radii / 10.)

plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(np.radians(theta), radii)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

#ax.bar(theta, radii, width=.1, bottom=0.0, alpha=0.5)

plt.show()


fig = plt.figure()
plt.bar(azimuthrange_rotated, np.concatenate((np.take(mean_cosdistdeg_az_RNN[x,],leftaxis),np.take(mean_cosdistdeg_az_RNN[x,],rightaxis))))
# create errorbar plot for all RNN models
fig = plt.figure()
for x in range(len(mean_cosdistdeg_az_RNN[:,1])):
    if not x == 4: # exclude the RNN model using drop-out for now
        plt.errorbar(azimuthrange_rotated, np.concatenate((np.take(mean_cosdistdeg_az_RNN[x,],leftaxis),np.take(mean_cosdistdeg_az_RNN[x,],rightaxis))), xerr = None, yerr = np.concatenate((np.take(sem_cosdistdeg_az_RNN[x,],leftaxis),np.take(sem_cosdistdeg_az_RNN[x,],rightaxis))), fmt = '.')
plt.legend(['119 nodes | AD loss', '119 nodes | MSE loss', '79 nodes | AD loss', '79 nodes | MSE loss', '99 nodes | AD loss', '99 nodes | MSE loss'], loc='upper right', fontsize = 12)
plt.xlabel('Target azimuth position (degrees)', fontsize = 12)
plt.ylabel('Angular distance (degrees)', fontsize = 12)
plt.grid(linestyle = ':', alpha = .7)
plt.ylim((0,40))

# create errorbar plot for all CNN models
fig = plt.figure()
for x in range(len(mean_cosdistdeg_az[:,1])):
    if x > 36: # this is where the subtract models starrt
        plt.errorbar(azimuthrange_rotated, np.concatenate((np.take(mean_cosdistdeg_az[x,],leftaxis),np.take(mean_cosdistdeg_az[x,],rightaxis))), xerr = None, yerr = np.concatenate((np.take(sem_cosdistdeg_az[x,],leftaxis),np.take(sem_cosdistdeg_az[x,],rightaxis))), fmt = '.')
#plt.legend(['119 nodes | AD loss', '119 nodes | MSE loss', '79 nodes | AD loss', '79 nodes | MSE loss', '99 nodes | AD loss', '99 nodes | MSE loss'], loc='upper right', fontsize = 12)
plt.xlabel('Target azimuth position (degrees)', fontsize = 12)
plt.ylabel('Angular distance (degrees)', fontsize = 12)
plt.grid(linestyle = ':', alpha = .7)
plt.ylim((0,40))
    
#==============================================================================
# create plots for the first group of CNN models 
#==============================================================================
# group specifications: largest number of kernels, concatenate, per kernel sizes
MergeMethod = '_S_'
KernelSpec = '64-64-64-128'
DropOutSpec = 'd2-d2-d2-d2'
LossSpec = 'MSE'
KernelSize = '17-17-17-17'

ModelIdx = np.ones((len(filenames_CNN),1), dtype = bool)
# get indices of these models from filenames using a boolean
for x in range(len(filenames_CNN)):
    ModelIdx[x] = MergeMethod in filenames_CNN[x] and KernelSpec in filenames_CNN[x] and DropOutSpec in filenames_CNN[x] and LossSpec in filenames_CNN[x] and KernelSize in filenames_CNN[x]
# turn into integer scalar array
ModelIdx = 1*ModelIdx
ModelsToPlot = np.where(ModelIdx == 1)
ModelsToPlot = np.asarray(ModelsToPlot)

# this is an example of a polar plot
theta_pred = np.squeeze(mean_predangles[ModelsToPlot[0,0],])
r_pred = np.ones(len(np.squeeze(mean_predangles[ModelsToPlot[0,0],])))
theta_true = azimuthrange
r_true = np.ones(len(azimuthrange))*1.5   
colors_pred = azimuthrange
colors_true = azimuthrange
fig = plt.figure()
ax = fig.add_subplot(111,projection = 'polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rorigin(-2.5)
ax.set_rlim(.5,2)
ax.set_yticklabels([])
ax.grid(linewidth = .75, linestyle = ':')
c = ax.scatter(np.radians(theta_pred), r_pred, c = colors_pred, cmap = 'jet')
c = ax.scatter(np.radians(theta_true), r_true, marker = '^', c = colors_true, cmap = 'jet', alpha = 1)

# this is to create the rectangular plots with predictions
anglecheck1 = 0
color1 = (1,0,0)
anglecheck2 = 90
color2 = (0.98, 0.95, 0.35)
anglecheck3 = 180
color3 = (0.62,1,0.24)
anglecheck4 = 270
color4 = (0.35,0.5,0.98)
plt.figure()
plt.scatter(models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck1),0],models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck2),0],models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck3),0],models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck4),0],models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Target locations (x) and predicted\nlocations (o) in Cartesian\ncoordinates',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)

anglecheck1 = 40
color1 = (147/255,248/255,254/255)
anglecheck2 = 140
color2 = (36/255,31/255,249/255)
anglecheck3 = 220
color3 = (218/255,43/255,200/255)
anglecheck4 = 320
color4 = (1,0,0)
plt.figure()
plt.scatter(models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck1),0],models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck2),0],models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck3),0],models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck4),0],models_predictions_CNN[ModelsToPlot[0,0],np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Target locations (x) and predicted \nlocations (o) in Cartesian\n coordinates',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)


#==============================================================================
# create plots for the first group of RNN models 
#==============================================================================
# group specifications: largest number of kernels, concatenate, per kernel sizes
NodeSpec = '119'
LossSpec = 'MSE'

ModelIdx = np.ones((len(filenames_RNN),1), dtype = bool)
# get indices of these models from filenames using a boolean
for x in range(RNN_nrofmodels):
    ModelIdx[x] = NodeSpec in filenames_RNN[x] and LossSpec in filenames_RNN[x]
# turn into integer scalar array
ModelIdx = 1*ModelIdx
ModelsToPlot = np.where(ModelIdx == 1)
ModelsToPlot = np.asarray(ModelsToPlot)

# this is an example of a polar plot
theta_pred = np.squeeze(mean_predangles_RNN[ModelsToPlot[0,0],])
r_pred = np.ones(len(np.squeeze(mean_predangles_RNN[ModelsToPlot[0,0],])))
theta_true = azimuthrange
r_true = np.ones(len(azimuthrange))*1.5   
colors_pred = azimuthrange
colors_true = azimuthrange
fig = plt.figure()
ax = fig.add_subplot(111,projection = 'polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rorigin(-2.5)
ax.set_rlim(.5,2)
ax.set_yticklabels([])
ax.grid(linewidth = .75, linestyle = ':')
c = ax.scatter(np.radians(theta_pred), r_pred, c = colors_pred, cmap = 'jet')
c = ax.scatter(np.radians(theta_true), r_true, marker = '^', c = colors_true, cmap = 'jet', alpha = 1)

# this is to create the rectangular plots with predictions
anglecheck1 = 0
color1 = (1,0,0)
anglecheck2 = 90
color2 = (0.98, 0.95, 0.35)
anglecheck3 = 180
color3 = (0.62,1,0.24)
anglecheck4 = 270
color4 = (0.35,0.5,0.98)
plt.figure()
plt.scatter(models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck1),0],models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck1),1],color=color1, alpha=0.4)
plt.scatter(labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck1),0],labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck2),0],models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck2),1],color=color2, alpha=0.4)
plt.scatter(labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck2),0],labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck3),0],models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck3),1],color=color3, alpha=0.4)
plt.scatter(labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck3),0],labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck4),0],models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck4),1],color=color4, alpha=0.4)
plt.scatter(labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck4),0],labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Target locations (x) and predicted\nlocations (o) in Cartesian\ncoordinates',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)

anglecheck1 = 40
color1 = (147/255,248/255,254/255)
anglecheck2 = 140
color2 = (36/255,31/255,249/255)
anglecheck3 = 220
color3 = (218/255,43/255,200/255)
anglecheck4 = 320
color4 = (1,0,0)
plt.figure()
plt.scatter(models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck1),0],models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck1),1],color=color1, alpha=0.4)
plt.scatter(labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck1),0],labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck2),0],models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck2),1],color=color2, alpha=0.4)
plt.scatter(labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck2),0],labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck3),0],models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck3),1],color=color3, alpha=0.4)
plt.scatter(labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck3),0],labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.scatter(models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck4),0],models_predictions_RNN[ModelsToPlot[0,0],np.squeeze(names_val_angle_RNN==anglecheck4),1],color=color4, alpha=0.4)
plt.scatter(labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck4),0],labels_val_RNN[np.squeeze(names_val_angle_RNN==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=100, edgecolors="k",linewidth=1)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Target locations (x) and predicted \nlocations (o) in Cartesian\n coordinates',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .1)
