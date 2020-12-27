# script to calculate values for CNN files
import numpy as np
import math


def Evaluation_ModelPredictions_CNN_calculations(names_val, models_predictions_CNN, labels_val, azimuthrange):
    
    
    names_val_angle = np.empty([len(names_val)])
    cnt1 = 0
    for x in names_val:
        names_val_angle[cnt1] = x[1:4]
        cnt1 += 1
    
    # retrieve whether it is anechoic or reverb, if reverb code 1
    names_val_env = np.zeros([len(names_val)])
    cnt2 = 0
    for x in names_val:
        if x.find('RI02') != -1:
            names_val_env[cnt2] = 1
        cnt2 += 1
    
    # compute euclidean distance
    dims_matrix_mse = np.shape(models_predictions_CNN)
    mse = np.empty([dims_matrix_mse[0],dims_matrix_mse[1]])
    for w in range(dims_matrix_mse[0]):
        for x in range(dims_matrix_mse[1]):
            mse[w,x] = np.mean(np.square(models_predictions_CNN[w,x] - labels_val[x]))    

    # compute cosine distance in degrees
    dims_matrix_cosdist = np.shape(models_predictions_CNN)
    cosine_distance_degrees = np.empty([dims_matrix_cosdist[0],dims_matrix_cosdist[1]])
    for w in range(dims_matrix_cosdist[0]):
        for  x in range(dims_matrix_cosdist[1]):
             cossim = np.sum(labels_val[x]*models_predictions_CNN[w,x])/(np.sqrt(np.sum(np.square(labels_val[x])))*np.sqrt(np.sum(np.square(models_predictions_CNN[w,x]))))
             cosine_distance_degrees[w,x] = np.arccos(cossim)*180/np.pi # this is the actual angle computed by multiplying the coefficient with 180 (max angular distance)
     #        cosine_distance_degrees[w,x] = np.arccos(cossim)/np.pi # this is the angular similarity coefficient bounded between 0 and 1

    # convert models_predictions_CNN to angle
    models_predictions_CNN_degs = np.zeros((len(models_predictions_CNN), len(models_predictions_CNN[0,:,:])))
    for w in range(len(models_predictions_CNN)):
        for x in range(len(models_predictions_CNN[0,:,:])):
            tempx = models_predictions_CNN[w,x,0]
            tempy = models_predictions_CNN[w,x,1]
            temp_pred_angle = math.degrees(math.atan(tempy/tempx))
            if tempx < 0 and tempy > 0:
                temp_pred_angle = np.abs(temp_pred_angle)+270 #left upper quadrant
            elif tempx < 0 and tempy < 0:
                temp_pred_angle = 270-temp_pred_angle
            elif tempx > 0 and tempy < 0:
                temp_pred_angle = 90+np.abs(temp_pred_angle)
            elif tempx > 0 and tempy > 0:
                temp_pred_angle = np.abs(temp_pred_angle-90)
            elif tempx == -1 and tempy == 0: # these are for exact values
                temp_pred_angle = 270
            elif tempx == 0 and tempy == -1:
                temp_pred_angle = 180
            elif tempx == 0 and tempy == 1:
                temp_pred_angle = 0
            elif tempx == 1 and tempy == 0:
                temp_pred_angle = 90
            models_predictions_CNN_degs[w,x] = temp_pred_angle
            
    # get labels per location
    models_predictions_CNN_degs_azloc = np.zeros((len(models_predictions_CNN_degs),len(azimuthrange),int(len(models_predictions_CNN_degs[0,:])/len(azimuthrange))))
    for w in range(len(models_predictions_CNN_degs)):
        for x in range(len(azimuthrange)):
            models_predictions_CNN_degs_azloc[w,x,:] = models_predictions_CNN_degs[w,np.where(names_val_angle==azimuthrange[x])]


    # now you want to compute the cosine distance as a function of azimuth location
    mean_mse_az =  np.empty([dims_matrix_mse[0],len(azimuthrange)])
    sem_mse_az = np.empty([dims_matrix_mse[0],len(azimuthrange)])
    mean_cosdistdeg_az = np.empty([dims_matrix_cosdist[0],len(azimuthrange)])
    stdev_cosdistdeg_az = np.empty([dims_matrix_cosdist[0],len(azimuthrange)])
    sem_cosdistdeg_az = np.empty([dims_matrix_cosdist[0],len(azimuthrange)])
    for w in range(dims_matrix_cosdist[0]):
        for x in range(len(azimuthrange)):
            mean_mse_az[w,x] = np.mean(np.squeeze(mse[w,np.where(names_val_angle==azimuthrange[x])]))
            sem_mse_az[w,x] =  np.std(np.squeeze(mse[w,np.where(names_val_angle==azimuthrange[x])]))/np.sqrt(len(np.squeeze(mse[w,np.where(names_val_angle==azimuthrange[x])])))
            mean_cosdistdeg_az[w,x] = np.mean(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x])]))
            stdev_cosdistdeg_az[w,x] = np.std(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x])]))
            sem_cosdistdeg_az[w,x] = np.std(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x])]))/np.sqrt(len(np.squeeze(cosine_distance_degrees[w,np.where(names_val_angle==azimuthrange[x])])))
    
    # now you want to compute the cosine distance as a function of env
    env_range = [0,1]
    mean_cosdistdeg_env = np.empty([dims_matrix_cosdist[0],len(env_range)]) # 2 for the number of environments that you have
    stdev_cosdistdeg_env = np.empty([dims_matrix_cosdist[0],len(env_range)])
    for w in range(dims_matrix_cosdist[0]):
        for x in range(len(env_range)):
            mean_cosdistdeg_env[w,x] = np.mean(np.squeeze(cosine_distance_degrees[w,np.where(names_val_env==env_range[x])]))
            stdev_cosdistdeg_env[w,x] = np.std(np.squeeze(cosine_distance_degrees[w,np.where(names_val_env==env_range[x])]))
    
    # compute mean and standard deviation of error per target azimuth location
    mean_prediction = np.empty([len(models_predictions_CNN),len(azimuthrange),2])
    stdev_prediction =  np.empty([len(models_predictions_CNN),len(azimuthrange),2])
    sem_prediction =  np.empty([len(models_predictions_CNN),len(azimuthrange),2])
    for w in range(len(models_predictions_CNN)):
        for x in range(len(azimuthrange)):
            mean_prediction[w,x] = np.mean(np.squeeze(models_predictions_CNN[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)
            stdev_prediction[w,x] = np.std(np.squeeze(models_predictions_CNN[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)
            sem_prediction[w,x] = np.std(np.squeeze(models_predictions_CNN[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)/np.sqrt(len(np.squeeze(models_predictions_CNN[w,np.where(names_val_angle==azimuthrange[x])])))
         
    # convert mean prediction to mean predicted angle
    mean_predangles = np.empty([len(models_predictions_CNN),len(azimuthrange)])
    for w in range(len(models_predictions_CNN)):
        for x in range(len(azimuthrange)):
            tempx = mean_prediction[w,x,0]
            tempy = mean_prediction[w,x,1]
            # to compute predicted angle
            temp_pred_angle = math.degrees(math.atan(tempy/tempx))
            # change reference back to zero at the front instead of at the right
            if tempx < 0 and tempy > 0:
                temp_pred_angle = np.abs(temp_pred_angle)+270 #left upper quadrant
            elif tempx < 0 and tempy < 0:
                temp_pred_angle = 270-temp_pred_angle
            elif tempx > 0 and tempy < 0:
                temp_pred_angle = 90+np.abs(temp_pred_angle)
            elif tempx > 0 and tempy > 0:
                temp_pred_angle = np.abs(temp_pred_angle-90)
            elif tempx == -1 and tempy == 0: # these are for exact values
                temp_pred_angle = 270
            elif tempx == 0 and tempy == -1:
                temp_pred_angle = 180
            elif tempx == 0 and tempy == 1:
                temp_pred_angle = 0
            elif tempx == 1 and tempy == 0:
                temp_pred_angle = 90
            mean_predangles[w,x] = temp_pred_angle
    
    
    # label per location (complicated way of retrieving it but OK)
    mean_label = np.empty([len(azimuthrange),2])
    for x in range(len(azimuthrange)):
        mean_label[x] =  np.mean(labels_val[np.where(names_val_angle==azimuthrange[x])],axis=0)
   
    
    CNN_dict_preds = dict(names_val_angle = names_val_angle, names_val_env = names_val_env, mse = mse, cosine_distance_degrees = cosine_distance_degrees, mean_mse_az = mean_mse_az, sem_mse_az = sem_mse_az,\
                          mean_cosdistdeg_az = mean_cosdistdeg_az, stdev_cosdistdeg_az = stdev_cosdistdeg_az, sem_cosdistdeg_az = sem_cosdistdeg_az, mean_cosdistdeg_env = mean_cosdistdeg_env, stdev_cosdistdeg_env = stdev_cosdistdeg_env,\
                          mean_prediction = mean_prediction, stdev_prediction = stdev_prediction, sem_prediction = sem_prediction, mean_predangles = mean_predangles, mean_label = mean_label, models_predictions_CNN_degs = models_predictions_CNN_degs,\
                          models_predictions_CNN_degs_azloc = models_predictions_CNN_degs_azloc, models_predictions_CNN_xy = models_predictions_CNN, labels_val = labels_val)
    
    return CNN_dict_preds


def Evaluation_ModelPredictions_CNN_calculations_reversalcorrected(names_val, models_predictions_CNN, labels_val, azimuthrange):
    
    #--------------------------------------------------------------------------
    # retrieve names and listening environments
    #--------------------------------------------------------------------------
    
    names_val_angle = np.empty([len(names_val)])
    cnt1 = 0
    for x in names_val:
        names_val_angle[cnt1] = x[1:4]
        cnt1 += 1
    
    # retrieve whether it is anechoic or reverb, if reverb code 1
    names_val_env = np.zeros([len(names_val)])
    cnt2 = 0
    for x in names_val:
        if x.find('RI02') != -1:
            names_val_env[cnt2] = 1
        cnt2 += 1

        
    #--------------------------------------------------------------------------
    # check which predictions are front-back reversals
    #--------------------------------------------------------------------------
    
    # convert mean prediction to mean predicted angle
    predangles_check = np.empty([len(models_predictions_CNN),len(models_predictions_CNN[0,:,0])])
    for w in range(len(models_predictions_CNN)):
        for x in range(len(models_predictions_CNN[0,:,0])):
            tempx = models_predictions_CNN[w,x,0]
            tempy = models_predictions_CNN[w,x,1]
            # to compute predicted angle
            temp_pred_angle = math.degrees(math.atan(tempy/tempx))
            # change reference back to zero at the front instead of at the right
            if tempx < 0 and tempy > 0:
                temp_pred_angle = np.abs(temp_pred_angle)+270 #left upper quadrant
            elif tempx < 0 and tempy < 0:
                temp_pred_angle = 270-temp_pred_angle
            elif tempx > 0 and tempy < 0:
                temp_pred_angle = 90+np.abs(temp_pred_angle)
            elif tempx > 0 and tempy > 0:
                temp_pred_angle = np.abs(temp_pred_angle-90)
            elif tempx == -1 and tempy == 0: # these are for exact values
                temp_pred_angle = 270
            elif tempx == 0 and tempy == -1:
                temp_pred_angle = 180
            elif tempx == 0 and tempy == 1:
                temp_pred_angle = 0
            elif tempx == 1 and tempy == 0:
                temp_pred_angle = 90
            predangles_check[w,x] = temp_pred_angle
            
    # correct data for front-back reversals
    reversals_check = np.zeros([len(predangles_check),len(predangles_check[0,:])], dtype=int)
    reversals_corrected_angle = np.zeros([len(predangles_check),len(predangles_check[0,:])]) # intialize matrix with uncorrected angles
    models_predictions_CNN_corrected = np.zeros(np.shape(models_predictions_CNN))
    for y in range(len(reversals_check)):
         for z in range(len(reversals_check[0,:])):
             # first check whether the prediction is in the back, otherwise you do not have to do this
             if predangles_check[y,z] > 90 and predangles_check[y,z] < 270:
                 # retrieve target angle
                 angle_target = names_val_angle[z]
                 # you only have to do the correction if angle_target is in the front, otherwise they are both in the back
                 if angle_target < 90 or angle_target > 270:
                     # check whether estimate is within criterion range of -15,+15 of the reference                      
                     # first flip estimate to the front
                     if angle_target < 181 and np.abs(angle_target-predangles_check[y,z]) > 15:
                         predangle_flipped = 180 - predangles_check[y,z]
                         # check whether the flipped estimate is within the [-15,+15] range
                         if predangle_flipped > angle_target - 20 and predangle_flipped < angle_target + 20:
                             reversals_check[y,z] = 1 # mark estimate as reversal
                             reversals_corrected_angle[y,z] = predangle_flipped # replace estimate with corrected estimate
                             models_predictions_CNN_corrected[y,z,0] = models_predictions_CNN[y,z,0] # leave x coordinate intact
                             models_predictions_CNN_corrected[y,z,1] = np.abs(models_predictions_CNN[y,z,1]) # flip y coordinate to the front
                         else:
                             reversals_corrected_angle[y,z] = predangles_check[y,z]
                             models_predictions_CNN_corrected[y,z,:] = models_predictions_CNN[y,z,:]
                     elif angle_target > 180 and np.abs(angle_target-predangles_check[y,z]) > 15:
                         predangle_flipped = 180 - predangles_check[y,z] + 360
                         # check whether the flipped estimate is within the [-15,+15] range
                         if predangle_flipped > angle_target - 20 and predangle_flipped < angle_target + 20:
                             reversals_check[y,z] = 1 # mark estimate as reversal
                             reversals_corrected_angle[y,z] = predangle_flipped # replace estimate with corrected estimate
                             models_predictions_CNN_corrected[y,z,0] = models_predictions_CNN[y,z,0] # leave x coordinate intact
                             models_predictions_CNN_corrected[y,z,1] = np.abs(models_predictions_CNN[y,z,1]) # flip y coordinate to the front
                         else:
                             reversals_corrected_angle[y,z] = predangles_check[y,z]
                             models_predictions_CNN_corrected[y,z,:] = models_predictions_CNN[y,z,:]
                     else: 
                         reversals_corrected_angle[y,z] = predangles_check[y,z]
                         models_predictions_CNN_corrected[y,z,:] = models_predictions_CNN[y,z,:]
                # if it's not a reversal, add the normal value to the array
                 else: 
                     reversals_corrected_angle[y,z] = predangles_check[y,z]
                     models_predictions_CNN_corrected[y,z,:] = models_predictions_CNN[y,z,:]
                # if it's not a reversal, add the normal value to the array
             else:
                 reversals_corrected_angle[y,z] = predangles_check[y,z]
                 models_predictions_CNN_corrected[y,z,:] = models_predictions_CNN[y,z,:]
         reversals_percentage = np.count_nonzero(reversals_check,axis=1)/len(reversals_check[0,])
            
   
    # compute euclidean distance
    dims_matrix_mse_corrected = np.shape(models_predictions_CNN_corrected)
    mse_corrected = np.empty([dims_matrix_mse_corrected[0],dims_matrix_mse_corrected[1]])
    for w in range(dims_matrix_mse_corrected[0]):
        for x in range(dims_matrix_mse_corrected[1]):
            mse_corrected[w,x] = np.mean(np.square(models_predictions_CNN_corrected[w,x] - labels_val[x]))    

    # compute cosine distance in degrees
    dims_matrix_cosdist_corrected = np.shape(models_predictions_CNN_corrected)
    cosine_distance_degrees_corrected = np.empty([dims_matrix_cosdist_corrected[0],dims_matrix_cosdist_corrected[1]])
    for w in range(dims_matrix_cosdist_corrected[0]):
        for  x in range(dims_matrix_cosdist_corrected[1]):
             cossim_corrected = np.sum(labels_val[x]*models_predictions_CNN_corrected[w,x])/(np.sqrt(np.sum(np.square(labels_val[x])))*np.sqrt(np.sum(np.square(models_predictions_CNN_corrected[w,x]))))
             cosine_distance_degrees_corrected[w,x] = np.arccos(cossim_corrected)*180/np.pi # this is the actual angle computed by multiplying the coefficient with 180 (max angular distance)
    
    # convert models_predictions_CNN to angle
    models_predictions_CNN_degs_corrected = np.zeros((len(models_predictions_CNN_corrected), len(models_predictions_CNN_corrected[0,:,:])))
    for w in range(len(models_predictions_CNN_corrected)):
        for x in range(len(models_predictions_CNN_corrected[0,:,:])):
            tempx = models_predictions_CNN_corrected[w,x,0]
            tempy = models_predictions_CNN_corrected[w,x,1]
            temp_pred_angle = math.degrees(math.atan(tempy/tempx))
            if tempx < 0 and tempy > 0:
                temp_pred_angle = np.abs(temp_pred_angle)+270 #left upper quadrant
            elif tempx < 0 and tempy < 0:
                temp_pred_angle = 270-temp_pred_angle
            elif tempx > 0 and tempy < 0:
                temp_pred_angle = 90+np.abs(temp_pred_angle)
            elif tempx > 0 and tempy > 0:
                temp_pred_angle = np.abs(temp_pred_angle-90)
            elif tempx == -1 and tempy == 0: # these are for exact values
                temp_pred_angle = 270
            elif tempx == 0 and tempy == -1:
                temp_pred_angle = 180
            elif tempx == 0 and tempy == 1:
                temp_pred_angle = 0
            elif tempx == 1 and tempy == 0:
                temp_pred_angle = 90
            models_predictions_CNN_degs_corrected[w,x] = temp_pred_angle

    # get labels per location
    models_predictions_CNN_degs_azloc_corrected = np.zeros((len(models_predictions_CNN_degs_corrected),len(azimuthrange),int(len(models_predictions_CNN_degs_corrected[0,:])/len(azimuthrange))))
    for w in range(len(models_predictions_CNN_degs_corrected)):
        for x in range(len(azimuthrange)):
            models_predictions_CNN_degs_azloc_corrected[w,x,:] = models_predictions_CNN_degs_corrected[w,np.where(names_val_angle==azimuthrange[x])]

    # now you want to compute the cosine distance as a function of azimuth location
    mean_mse_az_corrected =  np.empty([dims_matrix_mse_corrected[0],len(azimuthrange)])
    sem_mse_az_corrected = np.empty([dims_matrix_mse_corrected[0],len(azimuthrange)])
    mean_cosdistdeg_az_corrected = np.empty([dims_matrix_cosdist_corrected[0],len(azimuthrange)])
    stdev_cosdistdeg_az_corrected = np.empty([dims_matrix_cosdist_corrected[0],len(azimuthrange)])
    sem_cosdistdeg_az_corrected = np.empty([dims_matrix_cosdist_corrected[0],len(azimuthrange)])
    for w in range(dims_matrix_cosdist_corrected[0]):
        for x in range(len(azimuthrange)):
            mean_mse_az_corrected[w,x] = np.mean(np.squeeze(mse_corrected[w,np.where(names_val_angle==azimuthrange[x])]))
            sem_mse_az_corrected[w,x] =  np.std(np.squeeze(mse_corrected[w,np.where(names_val_angle==azimuthrange[x])]))/np.sqrt(len(np.squeeze(mse_corrected[w,np.where(names_val_angle==azimuthrange[x])])))
            mean_cosdistdeg_az_corrected[w,x] = np.mean(np.squeeze(cosine_distance_degrees_corrected[w,np.where(names_val_angle==azimuthrange[x])]))
            stdev_cosdistdeg_az_corrected[w,x] = np.std(np.squeeze(cosine_distance_degrees_corrected[w,np.where(names_val_angle==azimuthrange[x])]))
            sem_cosdistdeg_az_corrected[w,x] = np.std(np.squeeze(cosine_distance_degrees_corrected[w,np.where(names_val_angle==azimuthrange[x])]))/np.sqrt(len(np.squeeze(cosine_distance_degrees_corrected[w,np.where(names_val_angle==azimuthrange[x])])))
    
    # now you want to compute the cosine distance as a function of env
    env_range = [0,1]
    mean_cosdistdeg_env_corrected = np.empty([dims_matrix_cosdist_corrected[0],len(env_range)]) # 2 for the number of environments that you have
    stdev_cosdistdeg_env_corrected = np.empty([dims_matrix_cosdist_corrected[0],len(env_range)])
    for w in range(dims_matrix_cosdist_corrected[0]):
        for x in range(len(env_range)):
            mean_cosdistdeg_env_corrected[w,x] = np.mean(np.squeeze(cosine_distance_degrees_corrected[w,np.where(names_val_env==env_range[x])]))
            stdev_cosdistdeg_env_corrected[w,x] = np.std(np.squeeze(cosine_distance_degrees_corrected[w,np.where(names_val_env==env_range[x])]))
    
    # compute mean and standard deviation of error per target azimuth location
    mean_prediction_corrected = np.empty([len(models_predictions_CNN_corrected),len(azimuthrange),2])
    stdev_prediction_corrected =  np.empty([len(models_predictions_CNN_corrected),len(azimuthrange),2])
    sem_prediction_corrected =  np.empty([len(models_predictions_CNN_corrected),len(azimuthrange),2])
    for w in range(len(models_predictions_CNN_corrected)):
        for x in range(len(azimuthrange)):
            mean_prediction_corrected[w,x] = np.mean(np.squeeze(models_predictions_CNN_corrected[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)
            stdev_prediction_corrected[w,x] = np.std(np.squeeze(models_predictions_CNN_corrected[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)
            sem_prediction_corrected[w,x] = np.std(np.squeeze(models_predictions_CNN_corrected[w,np.where(names_val_angle==azimuthrange[x])]),axis=0)/np.sqrt(len(np.squeeze(models_predictions_CNN_corrected[w,np.where(names_val_angle==azimuthrange[x])])))
         
    # convert mean prediction to mean predicted angle
    mean_predangles_corrected = np.empty([len(models_predictions_CNN_corrected),len(azimuthrange)])
    for w in range(len(models_predictions_CNN_corrected)):
        for x in range(len(azimuthrange)):
            tempx = mean_prediction_corrected[w,x,0]
            tempy = mean_prediction_corrected[w,x,1]
            # to compute predicted angle
            temp_pred_angle_corrected = math.degrees(math.atan(tempy/tempx))
            # change reference back to zero at the front instead of at the right
            if tempx < 0 and tempy > 0:
                temp_pred_angle_corrected = np.abs(temp_pred_angle_corrected)+270 #left upper quadrant
            elif tempx < 0 and tempy < 0:
                temp_pred_angle_corrected = 270-temp_pred_angle_corrected
            elif tempx > 0 and tempy < 0:
                temp_pred_angle_corrected = 90+np.abs(temp_pred_angle_corrected)
            elif tempx > 0 and tempy > 0:
                temp_pred_angle_corrected = np.abs(temp_pred_angle_corrected-90)
            elif tempx == -1 and tempy == 0: # these are for exact values
                temp_pred_angle_corrected = 270
            elif tempx == 0 and tempy == -1:
                temp_pred_angle_corrected = 180
            elif tempx == 0 and tempy == 1:
                temp_pred_angle_corrected = 0
            elif tempx == 1 and tempy == 0:
                temp_pred_angle_corrected = 90
            mean_predangles_corrected[w,x] = temp_pred_angle_corrected

       
    # label per location (complicated way of retrieving it but OK)
    mean_label = np.empty([len(azimuthrange),2])
    for x in range(len(azimuthrange)):
        mean_label[x] =  np.mean(labels_val[np.where(names_val_angle==azimuthrange[x])],axis=0)
   
    CNN_dict_preds_revcorr = dict(names_val_angle = names_val_angle, names_val_env = names_val_env, mse_corrected = mse_corrected, cosine_distance_degrees_corrected = cosine_distance_degrees_corrected, mean_mse_az_corrected = mean_mse_az_corrected, sem_mse_az_corrected = sem_mse_az_corrected,\
                                  mean_cosdistdeg_az_corrected = mean_cosdistdeg_az_corrected, stdev_cosdistdeg_az_corrected = stdev_cosdistdeg_az_corrected, sem_cosdistdeg_az_corrected = sem_cosdistdeg_az_corrected, mean_cosdistdeg_env_corrected = mean_cosdistdeg_env_corrected, stdev_cosdistdeg_env_corrected = stdev_cosdistdeg_env_corrected,\
                                  mean_prediction_corrected = mean_prediction_corrected, stdev_prediction_corrected = stdev_prediction_corrected, sem_prediction_corrected = sem_prediction_corrected, mean_predangles_corrected = mean_predangles_corrected, mean_label = mean_label, reversals_percentage = reversals_percentage,\
                                  models_predictions_CNN_degs_corrected = models_predictions_CNN_degs_corrected, models_predictions_CNN_degs_azloc_corrected = models_predictions_CNN_degs_azloc_corrected, models_predictions_CNN_xy_corrected = models_predictions_CNN_corrected, labels_val = labels_val)
    
    return CNN_dict_preds_revcorr
    


    