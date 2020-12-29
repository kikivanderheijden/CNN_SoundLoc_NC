# script to visualize internal representations (feature maps, node activations)
import os

# specifications
dir_models = '/workspace/notebooks/histories'
dir_sounds = '/workspace/notebooks/sounds_small_npy/eval'
dir_scripts = '/workspace/notebooks/scripts'

os.chdir(dir_scripts)
# libraries
import numpy as np
import pickle
from tensorflow.keras import models # contains different types of models (use sequential model here?)
from tensorflow.keras.initializers import glorot_uniform
from CustLoss_MSE import cust_mean_squared_error # note that in this loss function, the axis of the MSE is set to 1
from CustLoss_cosine_distance_angular import cos_dist_2D_angular # note that in this loss function, the axis of the MSE is set to 1
from CustMet_cosine_distance_angular import cos_distmet_2D_angular

# load evaluation dataset
sounds_l = np.load(dir_sounds+'/an_l_eval_sounds.npy')
sounds_r = np.load(dir_sounds+'/an_r_eval_sounds.npy')
print('sound arrays loaded')

# specify model layers of interest (these are all the 2D conv layers), layer 2 = conv2D left, layer 3 = conv2D right
# layer 9 = conv2D on merge, layer 11 = conv2D , layer 14 = conv2D
layers_interest = [2,3,9,11,14]

# specify model names
con_mse = 'CNN_con_K-32-64-64-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_MSE_final.h5'
sub_mse = 'CNN_sub_K-32-32-64-128_KS-37-37-37-37_MP-12-22-22-32_DO-2-2-2-2-2_MSE_final.h5'
sum_mse = 'CNN_sum_K-32-64-64-128_KS-37-37-37-37_MP-12-22-22-32_DO-2-2-2-2-2_MSE_final.h5'
con_ad = 'CNN_con_K-32-64-128-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_AD_final.h5'
sub_ad = 'CNN_sub_K-32-32-64-128_KS-37-37-37-37_MP-12-22-22-32_DO-2-2-2-2-2_AD_final.h5'
sum_ad = 'CNN_sum_K-32-32-64-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_AD_final.h5'

# perform operation 6 times
model = models.load_model(dir_models+'/'+con_mse, custom_objects={'GlorotUniform': glorot_uniform(), "cust_mean_squared_error": cust_mean_squared_error, "cos_distmet_2D_angular": cos_distmet_2D_angular} ) 
# define a new model, input = testsound, output = intermediate representations for all layers in the previous model from the first
# successive_outputs = [layer.output for i in model.layers[0:]] # this defines outputs for all layers
successive_outputs = [model.layers[i].output for i in layers_interest]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)
feature_maps = visualization_model.predict([sounds_l,sounds_r])
pickle.dump(feature_maps, open(dir_models+'/featuremaps_con_mse.p','wb'))
print('model con mse done')

