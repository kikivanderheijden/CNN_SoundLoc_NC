#------------------------------------------------------------------------------
# Specifications
#------------------------------------------------------------------------------
# specify directories
dir_wrfiles = "/workspace/notebooks/models" # for testing on DSRI
#dir_wrfiles = r"C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning" # for testing locally

# import libraries
from tensorflow.keras import layers
from tensorflow.keras import models # contains different types of models (use sequential model here?)
from tensorflow.keras import optimizers # contains different types of back propagation algorithms to train the model, 
                                        # including sgd (stochastic gradient
#from CustLoss_MSE import cust_mean_squared_error # note that in this loss function, the axis of the MSE is set to 1
from CustLoss_cosine_distance_angular import cos_dist_2D_angular # note that in this loss function, the axis of the MSE is set to 1
from CustMet_cosine_distance_angular import cos_distmet_2D_angular

# specify parameters
modelname   = 'CNN_sum_K-32-32-64-128_KS-35-35-35-35_MP-12-22-22-32_DO-2-2-2-2-2_AD'
time_sound  = 750 # input dimension 1 (time)
nfreqs      = 99 # input dimension 2 (frequencies)

#------------------------------------------------------------------------------
# Define model architecture
#------------------------------------------------------------------------------
# CNN 1 - left channel
in1                 = layers.Input(shape=(time_sound,nfreqs,1)) # define input (rows, columns, channels (only one in my case))
model_l_conv1       = layers.Conv2D(32,(3,5),activation='relu', padding = 'same')(in1) # define first layer and input to the layer
model_l_conv1_mp    = layers.MaxPooling2D(pool_size = (1,2))(model_l_conv1)
model_l_conv1_mp_do = layers.Dropout(0.2)(model_l_conv1_mp)

# CNN 1 - right channel
in2                 = layers.Input(shape=(time_sound,nfreqs,1)) # define input
model_r_conv1       = layers.Conv2D(32,(3,5),activation='relu', padding = 'same')(in2) # define first layer and input to the layer
model_r_conv1_mp    = layers.MaxPooling2D(pool_size = (1,2))(model_r_conv1)
model_r_conv1_mp_do = layers.Dropout(0.2)(model_r_conv1_mp)

# CNN 2 - merged
model_final_merge       = layers.Add()([model_l_conv1_mp_do, model_r_conv1_mp_do]) 
model_final_conv1       = layers.Conv2D(32,(3,5),activation='relu', padding = 'same')(model_final_merge)
model_final_conv1_mp    = layers.MaxPooling2D(pool_size = (2,2))(model_final_conv1)
model_final_conv1_mp_do = layers.Dropout(0.2)(model_final_conv1_mp)

# CNN 3 - merged
model_final_conv2       = layers.Conv2D(64,(3,5), activation = 'relu', padding = 'same')(model_final_conv1_mp_do)
model_final_conv2_mp    = layers.MaxPooling2D(pool_size = (2,2))(model_final_conv2)
model_final_conv2_mp_do = layers.Dropout(0.2)(model_final_conv2_mp)

# CNN 4 - merged
model_final_conv3       = layers.Conv2D(128,(3,5), activation = 'relu', padding = 'same')(model_final_conv2_mp_do)
model_final_conv3_mp    = layers.MaxPooling2D(pool_size = (3,2))(model_final_conv3)
model_final_conv3_mp_do = layers.Dropout(0.2)(model_final_conv3_mp)

# flatten
model_final_flatten = layers.Flatten()(model_final_conv3_mp_do)
model_final_dropout = layers.Dropout(0.2)(model_final_flatten) # dropout for regularization
predicted_coords    = layers.Dense(2, activation = 'tanh')(model_final_dropout) # I have used the tanh activation because our outputs should be between -1 and 1

#------------------------------------------------------------------------------
# Create model
#------------------------------------------------------------------------------
# create
model = models.Model(inputs = [in1,in2], outputs = predicted_coords) # create
# compile
model.compile(loss = cos_dist_2D_angular, optimizer = optimizers.Adam(), metrics=['cosine_proximity','mse', cos_distmet_2D_angular])
# print summary
model.summary()
# save
model.save(dir_wrfiles+'/A_'+modelname+'.h5') # save model

