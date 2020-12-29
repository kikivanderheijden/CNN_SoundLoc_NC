# script to visualize internal representations (feature maps, node activations)
import os

# specifications
dir_models = '/workspace/notebooks/histories'
dir_sounds = '/workspace/notebooks/sounds_small_npy/eval'
dir_scripts = '/workspace/notebooks/scripts'
dir_featuremaps = '/workspace/notebooks/featuremaps'

os.chdir(dir_scripts)
# libraries
import numpy as np
from tensorflow.keras import models # contains different types of models (use sequential model here?)
from tensorflow.keras.initializers import glorot_uniform
from CustLoss_MSE import cust_mean_squared_error # note that in this loss function, the axis of the MSE is set to 1
from CustMet_cosine_distance_angular import cos_distmet_2D_angular

# load evaluation dataset
sounds_l = np.load(dir_sounds+'/an_l_eval_sounds.npy')
sounds_r = np.load(dir_sounds+'/an_r_eval_sounds.npy')
print('sound arrays loaded')

# divide sounds into four arrays to solve memory issues
sounds_l1 = sounds_l[0:3402,]
sounds_l2 = sounds_l[3402:6804,]
sounds_l3 = sounds_l[6804:10206,]
sounds_l4 = sounds_l[10206:,]
sounds_r1 = sounds_r[0:3402,]
sounds_r2 = sounds_r[3402:6804,]
sounds_r3 = sounds_r[6804:10206,]
sounds_r4 = sounds_r[10206:,]

del sounds_l, sounds_r
print('sound arrays deleted')

# specify model layers of interest (these are all the 2D conv layers), layer 2 = conv2D left, layer 3 = conv2D right
# layer 9 = conv2D on merge, layer 11 = conv2D , layer 14 = conv2D
layers_interest = [2,3,9,11,14]

# specify model name
con_mse = 'CNN_con_K-32-64-64-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_MSE_final.h5'

# perform operation 6 times
model = models.load_model(dir_models+'/'+con_mse, custom_objects={'GlorotUniform': glorot_uniform(), "cust_mean_squared_error": cust_mean_squared_error, "cos_distmet_2D_angular": cos_distmet_2D_angular} ) 
print('model loaded successfully')
# define a new model, input = testsound, output = intermediate representations for all layers in the previous model from the first
# successive_outputs = [layer.output for i in model.layers[0:]] # this defines outputs for all layers
successive_outputs = [model.layers[i].output for i in layers_interest]
visualization_model = models.Model(inputs = model.input, outputs = successive_outputs)

# predict first round
feature_maps = visualization_model.predict([sounds_l1,sounds_r1])
# retrieve individual feature maps 
fmap_l2_1 = feature_maps[0]
fmap_l3_1 = feature_maps[1]
fmap_l9_1 = feature_maps[0]
fmap_l11_1 = feature_maps[1]
fmap_l14_1 = feature_maps[1]
# save feature maps
np.save(dir_featuremaps+"/fmap_l2_1",fmap_l2_1)
np.save(dir_featuremaps+"/fmap_l3_1",fmap_l3_1)
np.save(dir_featuremaps+"/fmap_l9_1",fmap_l9_1)
np.save(dir_featuremaps+"/fmap_l11_1",fmap_l11_1)
np.save(dir_featuremaps+"/fmap_l14_1",fmap_l14_1)
del fmap_l2_1,fmap_l3_1,fmap_l9_1,fmap_l11_1,fmap_l14_1,sounds_l1,sounds_r1,feature_maps
print('model con mse1 done')

# predict second round
feature_maps = visualization_model.predict([sounds_l2,sounds_r2])
# retrieve individual feature maps 
fmap_l2_2 = feature_maps[0]
fmap_l3_2 = feature_maps[1]
fmap_l9_2 = feature_maps[0]
fmap_l11_2 = feature_maps[1]
fmap_l14_2 = feature_maps[1]
# save feature maps
np.save(dir_featuremaps+"/fmap_l2_2",fmap_l2_2)
np.save(dir_featuremaps+"/fmap_l3_2",fmap_l3_2)
np.save(dir_featuremaps+"/fmap_l9_2",fmap_l9_2)
np.save(dir_featuremaps+"/fmap_l11_2",fmap_l11_2)
np.save(dir_featuremaps+"/fmap_l14_2",fmap_l14_2)
del fmap_l2_2,fmap_l3_2,fmap_l9_2,fmap_l11_2,fmap_l14_2,sounds_l2,sounds_r2,feature_maps
print('model con mse2 done')


# predict third round
feature_maps = visualization_model.predict([sounds_l3,sounds_r3])
# retrieve individual feature maps 
fmap_l2_3 = feature_maps[0]
fmap_l3_3 = feature_maps[1]
fmap_l9_3 = feature_maps[0]
fmap_l11_3 = feature_maps[1]
fmap_l14_3 = feature_maps[1]
# save feature maps
np.save(dir_featuremaps+"/fmap_l2_3",fmap_l2_3)
np.save(dir_featuremaps+"/fmap_l3_3",fmap_l3_3)
np.save(dir_featuremaps+"/fmap_l9_3",fmap_l9_3)
np.save(dir_featuremaps+"/fmap_l11_3",fmap_l11_3)
np.save(dir_featuremaps+"/fmap_l14_3",fmap_l14_3)
del fmap_l2_3,fmap_l3_3,fmap_l9_3,fmap_l11_3,fmap_l14_3,sounds_l3,sounds_r3,feature_maps
print('model con mse3 done')

# predict fourth round
feature_maps = visualization_model.predict([sounds_l4,sounds_r4])
# retrieve individual feature maps 
fmap_l2_4 = feature_maps[0]
fmap_l3_4 = feature_maps[1]
fmap_l9_4 = feature_maps[0]
fmap_l11_4 = feature_maps[1]
fmap_l14_4 = feature_maps[1]
# save feature maps
np.save(dir_featuremaps+"/fmap_l2_4",fmap_l2_4)
np.save(dir_featuremaps+"/fmap_l3_4",fmap_l3_4)
np.save(dir_featuremaps+"/fmap_l9_4",fmap_l9_4)
np.save(dir_featuremaps+"/fmap_l11_4",fmap_l11_4)
np.save(dir_featuremaps+"/fmap_l14_4",fmap_l14_4)
del fmap_l2_4,fmap_l3_4,fmap_l9_4,fmap_l11_4,fmap_l14_4,sounds_l4,sounds_r4,feature_maps
print('model con mse3 done')


