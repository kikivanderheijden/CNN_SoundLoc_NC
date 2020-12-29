# script to visualize internal representations (feature maps, node activations)
import os

# specifications
dir_models = '/workspace/notebooks/histories'
dir_sounds = '/workspace/notebooks/sounds_small_npy/eval'
dir_scripts = '/workspace/notebooks/scripts'
dir_featuremaps = '/workspace/notebooks/featuremaps'

# local
#dir_models = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\NC_CNN_SoundLoc_EvaluationFiles'
#dir_sounds = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\NC_CNN_SoundLoc_EvaluationFiles'
#dir_scripts = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\NC_CNN_SoundLoc'
#dir_featuremaps = r'C:\Users\kiki.vanderheijden\Documents\PYTHON\NC_CNN_SoundLoc_EvaluationFiles'


os.chdir(dir_scripts)
# libraries
import pickle
import numpy as np
from tensorflow.keras import models # contains different types of models (use sequential model here?)
from tensorflow.keras.initializers import glorot_uniform
from CustLoss_cosine_distance_angular import cos_dist_2D_angular # note that in this loss function, the axis of the MSE is set to 1
from CustMet_cosine_distance_angular import cos_distmet_2D_angular

#-----------------------------------------------------------------------------
# specifications
#-----------------------------------------------------------------------------
# set azimuth range
azimuthrange = np.arange(0,360,10)
# specify model layers of interest (these are all the 2D conv layers), layer 2 = conv2D left, layer 3 = conv2D right
# layer 9 = conv2D on merge, layer 11 = conv2D , layer 14 = conv2D
layers_interest = [2,3,9,12,15]
# specify model name
sum_ad = 'CNN_sum_K-32-64-64-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_AD_final.h5'

#-----------------------------------------------------------------------------
# load and prepare data 
#-----------------------------------------------------------------------------

# load evaluation dataset
sounds_l = np.load(dir_sounds+'/an_l_eval_sounds.npy')
sounds_r = np.load(dir_sounds+'/an_r_eval_sounds.npy')
print('sound arrays loaded')

# load labels
labels_val =  np.load(dir_sounds+"/labels_eval_sounds.npy")
names_val = pickle.load(open(dir_sounds+'/listfilenames_eval_sounds.p','rb'))
# retrieve angle names
names_val_angle = np.empty([len(names_val)])
cnt1 = 0
for x in names_val:
    names_val_angle[cnt1] = x[1:4]
    cnt1 += 1

# load model
model = models.load_model(dir_models+'/'+sub_ad, custom_objects={'GlorotUniform': glorot_uniform(), "cos_dist_2D_angular": cos_dist_2D_angular, "cos_distmet_2D_angular": cos_distmet_2D_angular}) 
print('model loaded successfully')

#-----------------------------------------------------------------------------
# retrieve feature maps
#-----------------------------------------------------------------------------
# define a new model, input = testsound, output = intermediate representations for all layers in the previous model from the first
# successive_outputs = [layer.output for i in model.layers[0:]] # this defines outputs for all layers
successive_outputs = [model.layers[i].output for i in layers_interest]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)

# get indices of location
soundidxs_az = np.zeros((len(azimuthrange),int(np.shape(names_val)[0]/len(azimuthrange))),dtype = int)
for x in range(len(azimuthrange)):
   soundidxs_az[x] = np.where(names_val_angle==azimuthrange[x])[0]
print('location indices retrieved successfully')   
   
# get predictions per location, compute average
fmap_l2_avg = np.zeros((len(azimuthrange), np.shape(model.layers[2].output)[1], np.shape(model.layers[2].output)[2], np.shape(model.layers[2].output)[3]))
fmap_l3_avg = np.zeros((len(azimuthrange), np.shape(model.layers[3].output)[1], np.shape(model.layers[3].output)[2], np.shape(model.layers[3].output)[3]))
fmap_l9_avg = np.zeros((len(azimuthrange), np.shape(model.layers[9].output)[1], np.shape(model.layers[9].output)[2], np.shape(model.layers[9].output)[3]))
fmap_l12_avg = np.zeros((len(azimuthrange), np.shape(model.layers[12].output)[1], np.shape(model.layers[12].output)[2], np.shape(model.layers[12].output)[3]))
fmap_l15_avg = np.zeros((len(azimuthrange), np.shape(model.layers[15].output)[1], np.shape(model.layers[15].output)[2], np.shape(model.layers[15].output)[3]))
for x in range(len(azimuthrange)):
    feature_maps = visualization_model.predict([sounds_l[soundidxs_az[x],],sounds_r[soundidxs_az[x],]])
    #feature_maps = visualization_model.predict([sounds_l[1:3,],sounds_r[1:3,]])
    # retrieve individual feature maps 
    fmap_l2 = feature_maps[0]
    fmap_l3 = feature_maps[1]
    fmap_l9 = feature_maps[2]
    fmap_l12 = feature_maps[3]
    fmap_l15 = feature_maps[4]
    del feature_maps
    fmap_l2_avg[x,] = np.mean(fmap_l2,axis = 0)
    fmap_l3_avg[x,] = np.mean(fmap_l3,axis = 0)
    fmap_l9_avg[x,] = np.mean(fmap_l9,axis = 0)
    fmap_l12_avg[x,] = np.mean(fmap_l12,axis = 0)
    fmap_l15_avg[x,] = np.mean(fmap_l15,axis = 0)
    del fmap_l2, fmap_l3, fmap_l9, fmap_l12, fmap_l15
    print('one az location done')


# save feature maps
np.save(dir_featuremaps+"/sub_ad_fmap_l2_1",fmap_l2_avg)
np.save(dir_featuremaps+"/sub_ad_fmap_l3_1",fmap_l3_avg)
np.save(dir_featuremaps+"/sub_ad_fmap_l9_1",fmap_l9_avg)
np.save(dir_featuremaps+"/sub_ad_fmap_l12_1",fmap_l12_avg)
np.save(dir_featuremaps+"/sub_ad_fmap_l15_1",fmap_l15_avg)
del fmap_l2_avg,fmap_l3_avg,fmap_l9_avg,fmap_l12_avg,fmap_l15_avg
print('model sub_ad done')


