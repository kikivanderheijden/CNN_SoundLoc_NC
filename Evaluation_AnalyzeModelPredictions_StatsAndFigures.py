# script to compare model predictions

#------------------------------------------------------------------------------
# File locations 
#------------------------------------------------------------------------------

# set directories
dirfiles = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\NC_CNN_SoundLoc_EvaluationFiles'
dirscripts = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\NC_CNN_SoundLoc'
excelfile = r'C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning\DNN_HumanSoundLoc\DNN_modelspace_overview_performance.xlsx'

#------------------------------------------------------------------------------
# import libraries
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#import math
import scipy.stats
import statsmodels


#------------------------------------------------------------------------------
# Specifications
#------------------------------------------------------------------------------

# set azimuth range
azimuthrange = np.arange(0,360,10)
# best models
CNN_con_mse = "CNN_con_K-32-64-64-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_MSE"
CNN_con_ad = "CNN_con_K-32-64-128-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_AD"
CNN_sub_mse = "CNN_sub_K-32-32-64-128_KS-37-37-37-37_MP-12-22-22-32_DO-2-2-2-2-2_MSE"
CNN_sub_ad = "CNN_sub_K-32-32-64-128_KS-37-37-37-37_MP-12-22-22-32_DO-2-2-2-2-2_AD"
CNN_sum_mse = "CNN_sum_K-32-32-64-128_KS-37-37-37-37_MP-12-22-22-32_DO-2-2-2-2-2_MSE"
CNN_sum_ad = "CNN_sum_K-32-32-64-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_AD"

#------------------------------------------------------------------------------
# Load dictionaries
#------------------------------------------------------------------------------
CNN_dict_avgperf = np.load(dirfiles+'/CNN_dict_avgperf_2020-12-28.npy', allow_pickle=True)
CNN_dict_preds =  np.load(dirfiles+'/CNN_dict_preds_2020-12-28.npy', allow_pickle=True)
CNN_dict_preds_revcorr =  np.load(dirfiles+'/CNN_dict_preds_revcorr_2020-12-28.npy', allow_pickle=True)
# this is how to access the items CNN_dict_avgperf.item()['CNN_mse_names']

filenames_CNN = CNN_dict_preds.item()['filenames_CNN']
# remove unnecessary parts 
for x in range(len(filenames_CNN)):
    filenames_CNN[x] = filenames_CNN[x][1:-16]

#------------------------------------------------------------------------------
# Create figures
#------------------------------------------------------------------------------
# specifications 
darkblue = '#020884'
darkgreen = '#088402'


### bar graph of the percentage of front back reversals in the predictions for each of the three best models 
revperc = CNN_dict_preds_revcorr.item()['reversals_percentage'] # retrieve all reversal percentages
# create figure
plt.figure()
plt.grid(color = 'k', linestyle = ':', linewidth = .5)
plt.barh([6,5,4],[revperc[filenames_CNN.index(CNN_con_mse)]*100, revperc[filenames_CNN.index(CNN_sum_mse)]*100, revperc[filenames_CNN.index(CNN_sub_mse)]*100], color = darkblue, linewidth = 2, edgecolor = 'black')
plt.barh([3,2,1],[revperc[filenames_CNN.index(CNN_con_ad)]*100, revperc[filenames_CNN.index(CNN_sum_ad)]*100, revperc[filenames_CNN.index(CNN_sub_ad)]*100], color = darkgreen, linewidth = 2, edgecolor = 'black')
ylabels = ['Concatenate', 'Addition','Subtraction','Concatenate','Addition','Subtraction']
plt.yticks([6,5,4,3,2,1],ylabels,fontsize=20)
plt.xticks(fontsize = 20)

### polar plot of angular error with standard deviation
# for MSE model, concatenate 
mean_ad = CNN_dict_preds.item()['mean_cosdistdeg_az'][filenames_CNN.index(CNN_con_mse),]
mean_sem = CNN_dict_preds.item()['stdev_cosdistdeg_az'][filenames_CNN.index(CNN_con_mse),]
radii_mse_1 = mean_ad
radii_mse_2 = np.zeros(1)
radii_mse_2[0] = mean_ad[0]
radii_mse = np.concatenate((radii_mse_1, radii_mse_2)) # you have to add the last point at the end of the array to close the circle
err_mse_1 = mean_sem
err_mse_2 = np.zeros(1)
err_mse_2[0] = mean_sem[0]
err_mse =  np.concatenate((err_mse_1, err_mse_2))
plt.figure(figsize = (3,3))
ax = plt.subplot(111, projection='polar')
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1, theta_2))
ax.plot(np.radians(theta),radii_mse, color = darkblue, linewidth = 3)
ax.fill_between(np.radians(theta),radii_mse - np.radians(err_mse), radii_mse+np.radians(err_mse), color= darkblue, alpha = 0.3)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.grid(linestyle = ':', linewidth = 1)
ax.set_ylim(0,10)
ax.set_yticks(np.arange(0,10,2))
plt.title('MSE concatenation')
# for MSE model, addition
mean_ad = CNN_dict_preds.item()['mean_cosdistdeg_az'][filenames_CNN.index(CNN_sum_mse),]
mean_sem = CNN_dict_preds.item()['stdev_cosdistdeg_az'][filenames_CNN.index(CNN_sum_mse),]
radii_mse_1 = mean_ad
radii_mse_2 = np.zeros(1)
radii_mse_2[0] = mean_ad[0]
radii_mse = np.concatenate((radii_mse_1, radii_mse_2)) # you have to add the last point at the end of the array to close the circle
err_mse_1 = mean_sem
err_mse_2 = np.zeros(1)
err_mse_2[0] = mean_sem[0]
err_mse =  np.concatenate((err_mse_1, err_mse_2))
plt.figure(figsize = (3,3))
ax = plt.subplot(111, projection='polar')
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1, theta_2))
ax.plot(np.radians(theta),radii_mse, color = darkblue, linewidth = 3)
ax.fill_between(np.radians(theta),radii_mse - np.radians(err_mse), radii_mse+np.radians(err_mse), color= darkblue, alpha = 0.3)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.grid(linestyle = ':', linewidth = 1)
ax.set_ylim(0,10)
ax.set_yticks(np.arange(0,10,2))
plt.title('MSE addition')
# for MSE model, subtraction
mean_ad = CNN_dict_preds.item()['mean_cosdistdeg_az'][filenames_CNN.index(CNN_sub_mse),]
mean_sem = CNN_dict_preds.item()['stdev_cosdistdeg_az'][filenames_CNN.index(CNN_sub_mse),]
radii_mse_1 = mean_ad
radii_mse_2 = np.zeros(1)
radii_mse_2[0] = mean_ad[0]
radii_mse = np.concatenate((radii_mse_1, radii_mse_2)) # you have to add the last point at the end of the array to close the circle
err_mse_1 = mean_sem
err_mse_2 = np.zeros(1)
err_mse_2[0] = mean_sem[0]
err_mse =  np.concatenate((err_mse_1, err_mse_2))
plt.figure(figsize = (3,3))
ax = plt.subplot(111, projection='polar')
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1, theta_2))
ax.plot(np.radians(theta),radii_mse, color = darkblue, linewidth = 3)
ax.fill_between(np.radians(theta),radii_mse - np.radians(err_mse), radii_mse+np.radians(err_mse), color= darkblue, alpha = 0.3)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.grid(linestyle = ':', linewidth = 1)
ax.set_ylim(0,10)
ax.set_yticks(np.arange(0,10,2))
plt.title('MSE subtraction')
# for AD model, concatenate 
mean_ad = CNN_dict_preds.item()['mean_cosdistdeg_az'][filenames_CNN.index(CNN_con_ad),]
mean_sem = CNN_dict_preds.item()['stdev_cosdistdeg_az'][filenames_CNN.index(CNN_con_ad),]
radii_mse_1 = mean_ad
radii_mse_2 = np.zeros(1)
radii_mse_2[0] = mean_ad[0]
radii_mse = np.concatenate((radii_mse_1, radii_mse_2)) # you have to add the last point at the end of the array to close the circle
err_mse_1 = mean_sem
err_mse_2 = np.zeros(1)
err_mse_2[0] = mean_sem[0]
err_mse =  np.concatenate((err_mse_1, err_mse_2))
plt.figure(figsize = (3,3))
ax = plt.subplot(111, projection='polar')
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1, theta_2))
ax.plot(np.radians(theta),radii_mse, color = darkgreen, linewidth = 3)
ax.fill_between(np.radians(theta),radii_mse - np.radians(err_mse), radii_mse+np.radians(err_mse), color= darkgreen, alpha = 0.3)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.grid(linestyle = ':', linewidth = 1)
ax.set_ylim(0,10)
ax.set_yticks(np.arange(0,10,2))
plt.title('AD concatenation')
# for AD model, addition
mean_ad = CNN_dict_preds.item()['mean_cosdistdeg_az'][filenames_CNN.index(CNN_sum_ad),]
mean_sem = CNN_dict_preds.item()['stdev_cosdistdeg_az'][filenames_CNN.index(CNN_sum_ad),]
radii_mse_1 = mean_ad
radii_mse_2 = np.zeros(1)
radii_mse_2[0] = mean_ad[0]
radii_mse = np.concatenate((radii_mse_1, radii_mse_2)) # you have to add the last point at the end of the array to close the circle
err_mse_1 = mean_sem
err_mse_2 = np.zeros(1)
err_mse_2[0] = mean_sem[0]
err_mse =  np.concatenate((err_mse_1, err_mse_2))
plt.figure(figsize = (3,3))
ax = plt.subplot(111, projection='polar')
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1, theta_2))
ax.plot(np.radians(theta),radii_mse, color = darkgreen, linewidth = 3)
ax.fill_between(np.radians(theta),radii_mse - np.radians(err_mse), radii_mse+np.radians(err_mse), color= darkgreen, alpha = 0.3)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.grid(linestyle = ':', linewidth = 1)
ax.set_ylim(0,10)
ax.set_yticks(np.arange(0,10,2))
plt.title('AD addition')
# for AD model, subtraction
mean_ad = CNN_dict_preds.item()['mean_cosdistdeg_az'][filenames_CNN.index(CNN_sub_ad),]
mean_sem = CNN_dict_preds.item()['stdev_cosdistdeg_az'][filenames_CNN.index(CNN_sub_ad),]
radii_mse_1 = mean_ad
radii_mse_2 = np.zeros(1)
radii_mse_2[0] = mean_ad[0]
radii_mse = np.concatenate((radii_mse_1, radii_mse_2)) # you have to add the last point at the end of the array to close the circle
err_mse_1 = mean_sem
err_mse_2 = np.zeros(1)
err_mse_2[0] = mean_sem[0]
err_mse =  np.concatenate((err_mse_1, err_mse_2))
plt.figure(figsize = (3,3))
ax = plt.subplot(111, projection='polar')
theta_1 = azimuthrange
theta_2 = np.zeros(1)
theta_2[0] = azimuthrange[0]
theta = np.concatenate((theta_1, theta_2))
ax.plot(np.radians(theta),radii_mse, color = darkgreen, linewidth = 3)
ax.set_ylim(0,10)
ax.set_yticks(np.arange(0,10,2))
ax.fill_between(np.radians(theta),radii_mse - np.radians(err_mse), radii_mse+np.radians(err_mse), color= darkgreen, alpha = 0.3)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.grid(linestyle = ':', linewidth = 1)
plt.title('AD subtraction')


### line plot of MSE
# retrieve data
mean_mse = CNN_dict_preds.item()['mean_mse_az']
sem_mse = CNN_dict_preds.item()['sem_mse_az']
# plot data for MSE loss
plt.figure()
plt.errorbar(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_con_mse)],17),yerr = np.roll(sem_mse[filenames_CNN.index(CNN_con_mse)],17),  capsize = 2, linewidth = 2, fmt = 'none', color = darkblue)
plt.plot(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_con_mse)],17), color = darkblue,markersize = 10, linewidth = 4,  marker = 's')
plt.errorbar(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_sum_mse)],17),yerr = np.roll(sem_mse[filenames_CNN.index(CNN_sum_mse)],17), capsize = 2, linewidth = 2, linestyle = '--', fmt = 'none', color = darkblue)
plt.plot(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_sum_mse)],17), color = darkblue,  marker = '^',markersize = 10, linewidth = 4, linestyle = '--')
plt.errorbar(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_sub_mse)],17),yerr = np.roll(sem_mse[filenames_CNN.index(CNN_sub_mse)],17), capsize = 2, linewidth = 2, linestyle = ':', fmt = 'none', color = darkblue)
plt.plot(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_sub_mse)],17), color = darkblue, marker = 'o',markersize = 10, linewidth = 4, linestyle = ':')
plt.ylim(0,0.35)
plt.legend(['concatenation','addition','subtraction'], fontsize = 20)
# plot data for AD loss
plt.figure()
plt.errorbar(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_con_ad)],17),yerr = np.roll(sem_mse[filenames_CNN.index(CNN_con_ad)],17),  capsize = 2, linewidth = 2, fmt = 'none', color = darkgreen)
plt.plot(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_con_ad)],17), color = darkgreen, markersize = 10,linewidth = 4,  marker = 's')
plt.errorbar(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_sum_ad)],17),yerr = np.roll(sem_mse[filenames_CNN.index(CNN_sum_ad)],17), capsize = 2, linewidth = 2, linestyle = '--', fmt = 'none', color = darkgreen)
plt.plot(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_sum_ad)],17), color = darkgreen,  marker = '^', markersize = 10,linewidth = 4, linestyle = '--')
plt.errorbar(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_sub_ad)],17),yerr = np.roll(sem_mse[filenames_CNN.index(CNN_sub_ad)],17), capsize = 2, linewidth = 2, linestyle = ':', fmt = 'none', color = darkgreen)
plt.plot(np.linspace(1,36,36),np.roll(mean_mse[filenames_CNN.index(CNN_sub_ad)],17), color = darkgreen, marker = 'o', markersize = 10, linewidth = 4, linestyle = ':')
plt.ylim(0,0.35)
plt.legend(['concatenation','addition','subtraction'], fontsize = 20)

### this is to create the rectangular plots with predictions
# retrieve data
CNN_preds_con_mse = CNN_dict_preds.item()['models_predictions_CNN_xy'][filenames_CNN.index(CNN_con_mse)]
CNN_preds_sum_mse = CNN_dict_preds.item()['models_predictions_CNN_xy'][filenames_CNN.index(CNN_sum_mse)]
CNN_preds_sub_mse = CNN_dict_preds.item()['models_predictions_CNN_xy'][filenames_CNN.index(CNN_sub_mse)]
CNN_preds_con_ad = CNN_dict_preds.item()['models_predictions_CNN_xy'][filenames_CNN.index(CNN_con_ad)]
CNN_preds_sum_ad = CNN_dict_preds.item()['models_predictions_CNN_xy'][filenames_CNN.index(CNN_sum_ad)]
CNN_preds_sub_ad = CNN_dict_preds.item()['models_predictions_CNN_xy'][filenames_CNN.index(CNN_sub_ad)]
names_val_angle = CNN_dict_preds.item()['names_val_angle']
labels_val = CNN_dict_preds.item()['labels_val']

# specifications
anglecheck1 = 0
color1 = (246/255, 253/255, 0/255)
anglecheck11 = 30
color11 = (255/255,207/255,0/255)
anglecheck111 = 60
color111 = (255/255,154/255,0/255)
anglecheck2 = 90
color2 = (255/255, 89/255, 0/255)
anglecheck22 = 120
color22 = (255/255,0/255,30/255)
anglecheck222 = 150
color222 = (255/255,0/255,130/255)
anglecheck3 = 180
color3 = (227/255, 0/255, 255/255)
anglecheck33 = 210
color33 = (151/255,0/255,255/255)
anglecheck333 = 240
color333 = (103/255,0/255,255/255)
anglecheck4 = 270
color4 = (0/255, 51/255, 255/255)
anglecheck44 = 300
color44 = (0/255, 233/255, 255/255)
anglecheck444 = 330
color444 = (0/255, 255/255, 84/255)
# figure con mse
plt.figure()
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck1),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck1),1],color=color1, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck2),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck2),1],color=color2, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck3),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck3),1],color=color3, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck4),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck4),1],color=color4, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck11),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck11),1],color=color11, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck11),0],labels_val[np.squeeze(names_val_angle==anglecheck11),1],color=color11, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck22),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck22),1],color=color22, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck22),0],labels_val[np.squeeze(names_val_angle==anglecheck22),1],color=color22, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck33),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck33),1],color=color33, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck33),0],labels_val[np.squeeze(names_val_angle==anglecheck33),1],color=color33, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck44),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck44),1],color=color44, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck44),0],labels_val[np.squeeze(names_val_angle==anglecheck44),1],color=color44, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck111),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck111),1],color=color111, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck111),0],labels_val[np.squeeze(names_val_angle==anglecheck111),1],color=color111, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck222),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck222),1],color=color222, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck222),0],labels_val[np.squeeze(names_val_angle==anglecheck222),1],color=color222, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck333),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck333),1],color=color333, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck333),0],labels_val[np.squeeze(names_val_angle==anglecheck333),1],color=color333, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck444),0],CNN_preds_con_mse[np.squeeze(names_val_angle==anglecheck444),1],color=color444, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck444),0],labels_val[np.squeeze(names_val_angle==anglecheck444),1],color=color444, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Location predictions CNN con mse',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .5)
# figure sum mse
plt.figure()
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck1),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck1),1],color=color1, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck2),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck2),1],color=color2, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck3),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck3),1],color=color3, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck4),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck4),1],color=color4, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck11),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck11),1],color=color11, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck11),0],labels_val[np.squeeze(names_val_angle==anglecheck11),1],color=color11, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck22),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck22),1],color=color22, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck22),0],labels_val[np.squeeze(names_val_angle==anglecheck22),1],color=color22, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck33),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck33),1],color=color33, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck33),0],labels_val[np.squeeze(names_val_angle==anglecheck33),1],color=color33, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck44),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck44),1],color=color44, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck44),0],labels_val[np.squeeze(names_val_angle==anglecheck44),1],color=color44, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck111),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck111),1],color=color111, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck111),0],labels_val[np.squeeze(names_val_angle==anglecheck111),1],color=color111, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck222),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck222),1],color=color222, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck222),0],labels_val[np.squeeze(names_val_angle==anglecheck222),1],color=color222, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck333),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck333),1],color=color333, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck333),0],labels_val[np.squeeze(names_val_angle==anglecheck333),1],color=color333, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck444),0],CNN_preds_sum_mse[np.squeeze(names_val_angle==anglecheck444),1],color=color444, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck444),0],labels_val[np.squeeze(names_val_angle==anglecheck444),1],color=color444, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Location predictions CNN sum mse',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .5)
# figure sub mse
plt.figure()
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck1),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck1),1],color=color1, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck2),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck2),1],color=color2, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck3),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck3),1],color=color3, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck4),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck4),1],color=color4, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck11),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck11),1],color=color11, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck11),0],labels_val[np.squeeze(names_val_angle==anglecheck11),1],color=color11, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck22),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck22),1],color=color22, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck22),0],labels_val[np.squeeze(names_val_angle==anglecheck22),1],color=color22, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck33),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck33),1],color=color33, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck33),0],labels_val[np.squeeze(names_val_angle==anglecheck33),1],color=color33, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck44),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck44),1],color=color44, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck44),0],labels_val[np.squeeze(names_val_angle==anglecheck44),1],color=color44, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck111),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck111),1],color=color111, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck111),0],labels_val[np.squeeze(names_val_angle==anglecheck111),1],color=color111, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck222),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck222),1],color=color222, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck222),0],labels_val[np.squeeze(names_val_angle==anglecheck222),1],color=color222, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck333),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck333),1],color=color333, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck333),0],labels_val[np.squeeze(names_val_angle==anglecheck333),1],color=color333, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck444),0],CNN_preds_sub_mse[np.squeeze(names_val_angle==anglecheck444),1],color=color444, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck444),0],labels_val[np.squeeze(names_val_angle==anglecheck444),1],color=color444, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Location predictions CNN sub mse',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .5)
# figure con ad
plt.figure()
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck1),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck1),1],color=color1, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck2),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck2),1],color=color2, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck3),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck3),1],color=color3, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck4),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck4),1],color=color4, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck11),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck11),1],color=color11, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck11),0],labels_val[np.squeeze(names_val_angle==anglecheck11),1],color=color11, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck22),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck22),1],color=color22, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck22),0],labels_val[np.squeeze(names_val_angle==anglecheck22),1],color=color22, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck33),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck33),1],color=color33, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck33),0],labels_val[np.squeeze(names_val_angle==anglecheck33),1],color=color33, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck44),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck44),1],color=color44, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck44),0],labels_val[np.squeeze(names_val_angle==anglecheck44),1],color=color44, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck111),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck111),1],color=color111, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck111),0],labels_val[np.squeeze(names_val_angle==anglecheck111),1],color=color111, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck222),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck222),1],color=color222, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck222),0],labels_val[np.squeeze(names_val_angle==anglecheck222),1],color=color222, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck333),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck333),1],color=color333, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck333),0],labels_val[np.squeeze(names_val_angle==anglecheck333),1],color=color333, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck444),0],CNN_preds_con_ad[np.squeeze(names_val_angle==anglecheck444),1],color=color444, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck444),0],labels_val[np.squeeze(names_val_angle==anglecheck444),1],color=color444, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Location predictions CNN con ad',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .5)
# figure sum ad
plt.figure()
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck1),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck1),1],color=color1, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck2),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck2),1],color=color2, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck3),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck3),1],color=color3, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck4),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck4),1],color=color4, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck11),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck11),1],color=color11, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck11),0],labels_val[np.squeeze(names_val_angle==anglecheck11),1],color=color11, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck22),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck22),1],color=color22, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck22),0],labels_val[np.squeeze(names_val_angle==anglecheck22),1],color=color22, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck33),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck33),1],color=color33, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck33),0],labels_val[np.squeeze(names_val_angle==anglecheck33),1],color=color33, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck44),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck44),1],color=color44, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck44),0],labels_val[np.squeeze(names_val_angle==anglecheck44),1],color=color44, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck111),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck111),1],color=color111, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck111),0],labels_val[np.squeeze(names_val_angle==anglecheck111),1],color=color111, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck222),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck222),1],color=color222, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck222),0],labels_val[np.squeeze(names_val_angle==anglecheck222),1],color=color222, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck333),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck333),1],color=color333, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck333),0],labels_val[np.squeeze(names_val_angle==anglecheck333),1],color=color333, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck444),0],CNN_preds_sum_ad[np.squeeze(names_val_angle==anglecheck444),1],color=color444, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck444),0],labels_val[np.squeeze(names_val_angle==anglecheck444),1],color=color444, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Location predictions CNN sum ad',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .5)
# figure sub ad
plt.figure()
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck1),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck1),1],color=color1, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck1),0],labels_val[np.squeeze(names_val_angle==anglecheck1),1],color=color1, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck2),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck2),1],color=color2, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck2),0],labels_val[np.squeeze(names_val_angle==anglecheck2),1],color=color2, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck3),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck3),1],color=color3, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck3),0],labels_val[np.squeeze(names_val_angle==anglecheck3),1],color=color3, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck4),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck4),1],color=color4, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck4),0],labels_val[np.squeeze(names_val_angle==anglecheck4),1],color=color4, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck11),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck11),1],color=color11, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck11),0],labels_val[np.squeeze(names_val_angle==anglecheck11),1],color=color11, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck22),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck22),1],color=color22, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck22),0],labels_val[np.squeeze(names_val_angle==anglecheck22),1],color=color22, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck33),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck33),1],color=color33, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck33),0],labels_val[np.squeeze(names_val_angle==anglecheck33),1],color=color33, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck44),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck44),1],color=color44, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck44),0],labels_val[np.squeeze(names_val_angle==anglecheck44),1],color=color44, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck111),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck111),1],color=color111, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck111),0],labels_val[np.squeeze(names_val_angle==anglecheck111),1],color=color111, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck222),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck222),1],color=color222, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck222),0],labels_val[np.squeeze(names_val_angle==anglecheck222),1],color=color222, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck333),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck333),1],color=color333, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck333),0],labels_val[np.squeeze(names_val_angle==anglecheck333),1],color=color333, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=4)
plt.scatter(CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck444),0],CNN_preds_sub_ad[np.squeeze(names_val_angle==anglecheck444),1],color=color444, s = 200, edgecolors = "k", linewidth = 1, alpha=0.4)
plt.scatter(labels_val[np.squeeze(names_val_angle==anglecheck444),0],labels_val[np.squeeze(names_val_angle==anglecheck444),1],color=color444, alpha=.5, marker = "X",s=600, edgecolors="k",linewidth=2)
plt.axis('square')
plt.xlabel('x-coordinate',fontsize=15)
plt.ylabel('y-coordinate',fontsize=15)
plt.title('Location predictions CNN sub ad',fontweight = 'bold')
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid(color = 'k', linestyle = ':', linewidth = 1, alpha= .5)











#------------------------------------------------------------------------------
# Statistics
#------------------------------------------------------------------------------
# paired t-test for difference in error scores locations on the left vs locations on the right
# get data
mean_mse = CNN_dict_preds.item()['mean_mse_az']
sem_mse = CNN_dict_preds.item()['sem_mse_az']
mean_ad = CNN_dict_preds.item()['mean_cosdistdeg_az']
mean_sem = CNN_dict_preds.item()['stdev_cosdistdeg_az']
# for mse score
CNN_con_mse_Tmse = scipy.stats.ttest_rel(mean_mse[filenames_CNN.index(CNN_con_mse),1:18],mean_mse[filenames_CNN.index(CNN_con_mse),19:36])
CNN_sum_mse_Tmse = scipy.stats.ttest_rel(mean_mse[filenames_CNN.index(CNN_sum_mse),1:18],mean_mse[filenames_CNN.index(CNN_sum_mse),19:36])
CNN_sub_mse_Tmse = scipy.stats.ttest_rel(mean_mse[filenames_CNN.index(CNN_sub_mse),1:18],mean_mse[filenames_CNN.index(CNN_sub_mse),19:36])
CNN_con_ad_Tmse = scipy.stats.ttest_rel(mean_mse[filenames_CNN.index(CNN_con_ad),1:18],mean_mse[filenames_CNN.index(CNN_con_ad),19:36])
CNN_sum_ad_Tmse = scipy.stats.ttest_rel(mean_mse[filenames_CNN.index(CNN_sum_ad),1:18],mean_mse[filenames_CNN.index(CNN_sum_ad),19:36])
CNN_sub_ad_Tmse = scipy.stats.ttest_rel(mean_mse[filenames_CNN.index(CNN_sub_ad),1:18],mean_mse[filenames_CNN.index(CNN_sub_ad),19:36])
# correct p values for multiple comparisons
pvalcorr_Tmse = statsmodels.stats.multitest.multipletests([CNN_con_mse_Tmse[1],CNN_sum_mse_Tmse[1],CNN_sub_mse_Tmse[1],CNN_con_ad_Tmse[1],CNN_sum_ad_Tmse[1],CNN_sub_ad_Tmse[1]], method = 'fdr_bh')

# for ad score
CNN_con_mse_Tad = scipy.stats.ttest_rel(mean_ad[filenames_CNN.index(CNN_con_mse),1:18],mean_ad[filenames_CNN.index(CNN_con_mse),19:36])
CNN_sum_mse_Tad = scipy.stats.ttest_rel(mean_ad[filenames_CNN.index(CNN_sum_mse),1:18],mean_ad[filenames_CNN.index(CNN_sum_mse),19:36])
CNN_sub_mse_Tad = scipy.stats.ttest_rel(mean_ad[filenames_CNN.index(CNN_sub_mse),1:18],mean_ad[filenames_CNN.index(CNN_sub_mse),19:36])
CNN_con_ad_Tad = scipy.stats.ttest_rel(mean_ad[filenames_CNN.index(CNN_con_ad),1:18],mean_ad[filenames_CNN.index(CNN_con_ad),19:36])
CNN_sum_ad_Tad = scipy.stats.ttest_rel(mean_ad[filenames_CNN.index(CNN_sum_ad),1:18],mean_ad[filenames_CNN.index(CNN_sum_ad),19:36])
CNN_sub_ad_Tad = scipy.stats.ttest_rel(mean_ad[filenames_CNN.index(CNN_sub_ad),1:18],mean_ad[filenames_CNN.index(CNN_sub_ad),19:36])
# correct p values for multiple comparisons
pvalcorr_Tad = statsmodels.stats.multitest.multipletests([CNN_con_mse_Tad[1],CNN_sum_mse_Tad[1],CNN_sub_mse_Tad[1],CNN_con_ad_Tad[1],CNN_sum_ad_Tad[1],CNN_sub_ad_Tad[1]], method = 'fdr_bh')

# create bar graph of this
# for ad error
righterrors = [np.mean(mean_ad[filenames_CNN.index(CNN_con_mse),1:18]),np.mean(mean_ad[filenames_CNN.index(CNN_sum_mse),1:18]),np.mean(mean_ad[filenames_CNN.index(CNN_sub_mse),1:18]),np.mean(mean_ad[filenames_CNN.index(CNN_con_ad),1:18]),np.mean(mean_ad[filenames_CNN.index(CNN_sum_ad),1:18]),np.mean(mean_ad[filenames_CNN.index(CNN_sub_ad),1:18])]
lefterrors = [np.mean(mean_ad[filenames_CNN.index(CNN_con_mse),19:36]),np.mean(mean_ad[filenames_CNN.index(CNN_sum_mse),19:36]),np.mean(mean_ad[filenames_CNN.index(CNN_sub_mse),19:36]),np.mean(mean_ad[filenames_CNN.index(CNN_con_ad),19:36]),np.mean(mean_ad[filenames_CNN.index(CNN_sum_ad),19:36]),np.mean(mean_ad[filenames_CNN.index(CNN_sub_ad),19:36])]
righterror_sem = [np.std(mean_ad[filenames_CNN.index(CNN_con_mse),1:18])/17,np.std(mean_ad[filenames_CNN.index(CNN_sum_mse),1:18])/17,np.std(mean_ad[filenames_CNN.index(CNN_sub_mse),1:18])/17,np.std(mean_ad[filenames_CNN.index(CNN_con_ad),1:18])/17,np.std(mean_ad[filenames_CNN.index(CNN_sum_ad),1:18])/17,np.std(mean_ad[filenames_CNN.index(CNN_sub_ad),1:18])/17]
lefterror_sem = [np.std(mean_ad[filenames_CNN.index(CNN_con_mse),19:36])/17,np.std(mean_ad[filenames_CNN.index(CNN_sum_mse),19:36])/17,np.std(mean_ad[filenames_CNN.index(CNN_sub_mse),19:36])/17,np.std(mean_ad[filenames_CNN.index(CNN_con_ad),19:36])/17,np.std(mean_ad[filenames_CNN.index(CNN_sum_ad),19:36])/17,np.std(mean_ad[filenames_CNN.index(CNN_sub_ad),19:36])/17]
labels = ['con','sum','sub','con','sum','sub']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, lefterrors, width, label='Left', linewidth = 2,edgecolor = 'black')
rects2 = ax.bar(x + width/2, righterrors, width, label='Right', linewidth = 2,edgecolor = 'black')
rects3 = ax.errorbar([-0.16, 0.84 ,1.84 ,2.84,3.84,4.84],lefterrors,yerr= lefterror_sem, fmt = 'none',color = 'black' , capsize = 2, linewidth = 2)
rects4 = ax.errorbar([0.16, 1.16 ,2.16 ,3.16,4.16,5.16],righterrors,yerr= righterror_sem, fmt = 'none',color = 'black' , capsize = 2, linewidth = 2)
plt.title('AD error per hemifield')
plt.legend(['Left','Right'], fontsize = 30, loc = 'lower right')
# for mse error
righterrors = [np.mean(mean_mse[filenames_CNN.index(CNN_con_mse),1:18]),np.mean(mean_mse[filenames_CNN.index(CNN_sum_mse),1:18]),np.mean(mean_mse[filenames_CNN.index(CNN_sub_mse),1:18]),np.mean(mean_mse[filenames_CNN.index(CNN_con_ad),1:18]),np.mean(mean_mse[filenames_CNN.index(CNN_sum_ad),1:18]),np.mean(mean_mse[filenames_CNN.index(CNN_sub_ad),1:18])]
lefterrors = [np.mean(mean_mse[filenames_CNN.index(CNN_con_mse),19:36]),np.mean(mean_mse[filenames_CNN.index(CNN_sum_mse),19:36]),np.mean(mean_mse[filenames_CNN.index(CNN_sub_mse),19:36]),np.mean(mean_mse[filenames_CNN.index(CNN_con_ad),19:36]),np.mean(mean_mse[filenames_CNN.index(CNN_sum_ad),19:36]),np.mean(mean_mse[filenames_CNN.index(CNN_sub_ad),19:36])]
righterror_sem = [np.std(mean_mse[filenames_CNN.index(CNN_con_mse),1:18])/17,np.std(mean_mse[filenames_CNN.index(CNN_sum_mse),1:18])/17,np.std(mean_mse[filenames_CNN.index(CNN_sub_mse),1:18])/17,np.std(mean_mse[filenames_CNN.index(CNN_con_ad),1:18])/17,np.std(mean_mse[filenames_CNN.index(CNN_sum_ad),1:18])/17,np.std(mean_mse[filenames_CNN.index(CNN_sub_ad),1:18])/17]
lefterror_sem = [np.std(mean_mse[filenames_CNN.index(CNN_con_mse),19:36])/17,np.std(mean_mse[filenames_CNN.index(CNN_sum_mse),19:36])/17,np.std(mean_mse[filenames_CNN.index(CNN_sub_mse),19:36])/17,np.std(mean_mse[filenames_CNN.index(CNN_con_ad),19:36])/17,np.std(mean_mse[filenames_CNN.index(CNN_sum_ad),19:36])/17,np.std(mean_mse[filenames_CNN.index(CNN_sub_ad),19:36])/17]
labels = ['con','sum','sub','con','sum','sub']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, lefterrors, width, label='Left', linewidth = 2,edgecolor = 'black')
rects2 = ax.bar(x + width/2, righterrors, width, label='Right', linewidth = 2,edgecolor = 'black')
rects3 = ax.errorbar([-0.16, 0.84 ,1.84 ,2.84,3.84,4.84],lefterrors,yerr= lefterror_sem, fmt = 'none',color = 'black' , capsize = 2, linewidth = 2)
rects4 = ax.errorbar([0.16, 1.16 ,2.16 ,3.16,4.16,5.16],righterrors,yerr= righterror_sem, fmt = 'none',color = 'black' , capsize = 2, linewidth = 2)
plt.title('MSE error per hemifield')
plt.legend(['Left','Right'], fontsize = 30, loc = 'upper left')




































## confusion matrices of predictions
# CNN models original data
# cm_tarlocs = np.arange(0,370,10)
cm_tarlocs = [0,5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195,205,215,225,235,245,255,265,275,285,295,305,315,325,335,345,355,360]
CNN_cm_preds = CNN_dict_preds.item()['models_predictions_CNN_degs_azloc']
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
