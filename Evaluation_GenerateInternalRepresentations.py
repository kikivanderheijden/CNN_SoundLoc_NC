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
sounds_l = np.load(dir_sounds+'\\an_l_eval_sounds.npy')
sounds_r = np.load(dir_sounds+'\\an_r_eval_sounds.npy')
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

model = models.load_model(dir_models+'/'+sum_mse, custom_objects={'GlorotUniform': glorot_uniform(), "cust_mean_squared_error": cust_mean_squared_error, "cos_distmet_2D_angular": cos_distmet_2D_angular} ) 
# define a new model, input = testsound, output = intermediate representations for all layers in the previous model from the first
# successive_outputs = [layer.output for i in model.layers[0:]] # this defines outputs for all layers
successive_outputs = [model.layers[i].output for i in layers_interest]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)
feature_maps = visualization_model.predict([sounds_l,sounds_r])
pickle.dump(feature_maps, open(dir_models+'/featuremaps_sum_mse.p','wb'))
print('model sum mse done')

model = models.load_model(dir_models+'/'+sub_mse, custom_objects={'GlorotUniform': glorot_uniform(), "cust_mean_squared_error": cust_mean_squared_error, "cos_distmet_2D_angular": cos_distmet_2D_angular} ) 
# define a new model, input = testsound, output = intermediate representations for all layers in the previous model from the first
# successive_outputs = [layer.output for i in model.layers[0:]] # this defines outputs for all layers
successive_outputs = [model.layers[i].output for i in layers_interest]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)
feature_maps = visualization_model.predict([sounds_l,sounds_r])
pickle.dump(feature_maps, open(dir_models+'/featuremaps_sub_mse.p','wb'))
print('model sub mse done')

model = models.load_model(dir_models+'/'+con_ad,custom_objects={'GlorotUniform': glorot_uniform(), "cos_dist_2D_angular": cos_dist_2D_angular, "cos_distmet_2D_angular": cos_distmet_2D_angular}) 
# define a new model, input = testsound, output = intermediate representations for all layers in the previous model from the first
# successive_outputs = [layer.output for i in model.layers[0:]] # this defines outputs for all layers
successive_outputs = [model.layers[i].output for i in layers_interest]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)
feature_maps = visualization_model.predict([sounds_l,sounds_r])
pickle.dump(feature_maps, open(dir_models+'/featuremaps_con_ad.p','wb'))
print('model con ad done')

model = models.load_model(dir_models+'/'+sum_ad,custom_objects={'GlorotUniform': glorot_uniform(), "cos_dist_2D_angular": cos_dist_2D_angular, "cos_distmet_2D_angular": cos_distmet_2D_angular}) 
# define a new model, input = testsound, output = intermediate representations for all layers in the previous model from the first
# successive_outputs = [layer.output for i in model.layers[0:]] # this defines outputs for all layers
successive_outputs = [model.layers[i].output for i in layers_interest]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)
feature_maps = visualization_model.predict([sounds_l,sounds_r])
pickle.dump(feature_maps, open(dir_models+'/featuremaps_sum_ad.p','wb'))
print('model sum ad done')

model = models.load_model(dir_models+'/'+sub_ad,custom_objects={'GlorotUniform': glorot_uniform(), "cos_dist_2D_angular": cos_dist_2D_angular, "cos_distmet_2D_angular": cos_distmet_2D_angular}) 
# define a new model, input = testsound, output = intermediate representations for all layers in the previous model from the first
# successive_outputs = [layer.output for i in model.layers[0:]] # this defines outputs for all layers
successive_outputs = [model.layers[i].output for i in layers_interest]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)
feature_maps = visualization_model.predict([sounds_l,sounds_r])
pickle.dump(feature_maps, open(dir_models+'/featuremaps_sub_ad.p','wb'))
print('model sub ad done')

###### old code for visualization
# Retrieve are the names of the layers, so can have them as part of our plot
# layer_names = [layer.name for layer in model.layers]
# for layer_name, feature_map in zip(layer_names, successive_feature_maps):
#  print(feature_map.shape)
#  if len(feature_map.shape) == 4:
    
    # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
   
#    n_features = feature_map.shape[-1]  # number of features in the feature map
#    size_1       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
#    size_2       = feature_map.shape[2]   
    # We will tile our images in this matrix
#    display_grid = np.zeros((size_1, size_2 * n_features))
    
    # Postprocess the feature to be visually palatable
#    for i in range(n_features):
#      x  = feature_map[0, :, :, i]
      #x -= x.mean()
      #x /= x.std ()
      #x *=  64
      #x += 128
      #x  = np.clip(x, 0, 255).astype('uint8')
      # Tile each filter into a horizontal grid
#      display_grid[:, i * size_2 : (i + 1) * size_2] = x
# Display the grid
#    scale = 20. / n_features
#    plt.figure( figsize=(scale * n_features, scale) )
#    plt.title ( "tada" )
#    plt.grid  ( False )
#    plt.imshow( display_grid, aspect='auto', cmap='viridis' )
