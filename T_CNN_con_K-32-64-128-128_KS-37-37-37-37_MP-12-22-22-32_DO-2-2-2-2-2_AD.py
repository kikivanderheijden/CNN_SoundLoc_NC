# -----------------------------------------------------------------------------
# Specifications
# -----------------------------------------------------------------------------


# this is for the GPU
dir_mofiles = "/workspace/notebooks/models" # specify directory where model files are located 
dir_anfiles = "/workspace/notebooks/sounds_small_npy/train"


# import packages, libraries, functions to call them later
import numpy as np
#from pytictoc import TicToc
#t = TicToc()
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform
from CustLoss_cosine_distance_angular import cos_dist_2D_angular # note that in this loss function, the axis of the MSE is set to 1
from CustMet_cosine_distance_angular import cos_distmet_2D_angular
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# define name of model
modelname = "CNN_con_K-32-64-128-128_KS-37-37-37-37_MP-12-22-22-32_DO-2-2-2-2-2_AD"

# define training parameters
nrepochs = 200
sizebatches = 64

# define callback log functions
csv_loss_logger     = CSVLogger('history_'+modelname+'.csv')
dir_model_logger    = dir_mofiles+"/"+modelname+"_epoch_{epoch:02d}_val_loss-{val_loss:.2f}.h5"
model_logger        = ModelCheckpoint(dir_model_logger,  monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
stop_logger         = EarlyStopping(monitor= 'val_loss', min_delta=0.005, patience = 10, mode = 'min', baseline=None, restore_best_weights=True )
# all callbacks together
allcallbacks = [csv_loss_logger,model_logger,stop_logger]

#------------------------------------------------------------------------------
# Preparations
#------------------------------------------------------------------------------

# load data
labels_rand_train = np.load(dir_anfiles+"/labels_train_sounds.npy")
labels_rand_test = np.load(dir_anfiles+"/labels_test_sounds.npy")
an_l_rand_train = np.load(dir_anfiles+"/an_l_train_sounds.npy")
an_l_rand_test = np.load(dir_anfiles+"/an_l_test_sounds.npy")
an_r_rand_train = np.load(dir_anfiles+"/an_r_train_sounds.npy")
an_r_rand_test = np.load(dir_anfiles+"/an_r_test_sounds.npy")
print("Loading arrays completed")

# load model
#t.tic()
mymodel = load_model(dir_mofiles+"/A_"+modelname+".h5",custom_objects={'GlorotUniform': glorot_uniform(), "cos_dist_2D_angular": cos_dist_2D_angular, "cos_distmet_2D_angular": cos_distmet_2D_angular})
mymodel.summary()
#t.toc("loading the model took ")
print("Loading model completed")

#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------

# train the model
#t.tic()
history = mymodel.fit([an_l_rand_train, an_r_rand_train], labels_rand_train, validation_data=((an_l_rand_test,an_r_rand_test),labels_rand_test), epochs = nrepochs, batch_size = sizebatches, verbose = 1, use_multiprocessing = True, callbacks = allcallbacks)
#t.toc("training the model took ")

mymodel.save(modelname+"_final.h5")

print("Final model was saved")


#------------------------------------------------------------------------------
# OLD CODE
#------------------------------------------------------------------------------

# =============================================================================
# labels_rand_train = np.load(dir_anfiles+"/labels_train.npy")
# labels_rand_test = np.load(dir_anfiles+"/labels_test.npy")
# an_l_rand_train = np.load(dir_anfiles+"/an_l_train.npy")
# an_l_rand_test = np.load(dir_anfiles+"/an_r_test.npy")
# an_r_rand_train = np.load(dir_anfiles+"/an_l_train.npy")
# an_r_rand_test = np.load(dir_anfiles+"/an_r_test.npy")
# 
# an_l_rand_train = np.expand_dims(an_l_rand_train,axis = 3)
# an_l_rand_test = np.expand_dims(an_l_rand_test,axis = 3)
# an_r_rand_train = np.expand_dims(an_r_rand_train,axis = 3)
# an_r_rand_test = np.expand_dims(an_r_rand_test,axis = 3)
# 
# =============================================================================



# =============================================================================
# # metrics to save from the model
# hist_csv_file = 'history_model3_soundssmall.csv'
# with open(hist_csv_file, mode='w') as f:
#     history.to_csv(f)
# =============================================================================

# =============================================================================
#from tensorflow.keras.models import model_from_json
# t.tic()
# json_file = open(dir_mofiles+'/DNN_model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.summary()
# t.toc("loading the model took")
# =============================================================================
