
# read and preprocess sounds in the DSRI environment

# define directories, add r before the name because a normal string cannot be used as a path, alternatives are 
# using / or \\ instead of \
dir_anfiles = "/workspace/notebooks/sounds_small_npy/eval" # directory for both channels
#dir_anfiles = "C:/Users/kiki.vanderheijden/Documents/PYTHON/DataFiles" # sounds left channel

# import packages and libraries
import numpy as np
#from sklearn.utils import shuffle
from pytictoc import TicToc 
t = TicToc() # create instant of class
import pickle
import math

# set parameters
testset = 0 # size of testset
nrazlocs = 36 # number of azimuth locations
azimuthrange = np.arange(0,360,10)

t.tic()
# =============================================================================
# THESE ARE THE REAL ONES
# load numpy arrays from disk
an_l = np.load(dir_anfiles+"/an_l_sounds.npy")
an_r = np.load(dir_anfiles+"/an_r_sounds.npy")
labels = np.load(dir_anfiles+"/labels_sounds.npy")
filenames = pickle.load(open(dir_anfiles+'/listfilenames_sounds.p','rb'))
t.toc("loading the numpy arrays took ")
# =============================================================================

# retrieve labels from names
names_val_angle = np.empty([len(filenames)])
cnt1 = 0
for x in filenames:
    names_val_angle[cnt1] =int(x[1:4])
    cnt1 += 1

#cycle through the labels to get the indices of the angles
indices_val_angle = np.empty([nrazlocs,int(len(filenames)/nrazlocs)])
for x in range(nrazlocs):
    indices_val_angle[x,] = np.where(names_val_angle == azimuthrange[x])[0]

# compute overall number of sounds per location
nrsounds_loc = math.floor(len(labels[:,0])/nrazlocs)

# intiate empty matrics (or zero matrices)
an_l_eval = np.zeros((nrazlocs*math.floor((1-testset)*nrsounds_loc),len(an_l[1,:,:]),len(an_l[1,1,:]))) # round down here because train_test_split rounds the train size down
an_r_eval = np.zeros((nrazlocs*math.floor((1-testset)*nrsounds_loc),len(an_r[1,:,:]),len(an_r[1,1,:])))
labels_eval = np.zeros((nrazlocs*math.floor((1-testset)*nrsounds_loc),len(labels[1,:])))
filenames_eval = list()

# first take a part away for evaluation
for x in range(nrazlocs) :
    #  now take all those sounds --> you don't need to shuffle them because os.scandir already read them in an arbitrary order
    temp_idx = indices_val_angle[x,]
    temp_idx = temp_idx.astype(int)
    an_l_temp = an_l[temp_idx,]
    an_r_temp = an_r[temp_idx,]
    labels_temp = labels[temp_idx,]
    filenames_temp = [filenames[i] for i in temp_idx] # indexing from a list is a bit cumbersome...
    # add to the matrices in the correct positions
    an_l_eval[x*len(an_l_temp):(x+1)*len(an_l_temp)] = an_l_temp
    an_r_eval[x*len(an_r_temp):(x+1)*len(an_r_temp)] = an_r_temp
    labels_eval[x*len(labels_temp):(x+1)*len(labels_temp)] = labels_temp
    filenames_eval.extend(filenames_temp)

# add a fourth dimension ('channel') to train_an_l and train_an_r which should be 1, this is needed for the input to the DNN
an_l_eval = np.expand_dims(an_l_eval,axis = 3)
an_r_eval = np.expand_dims(an_r_eval,axis = 3)

#save numpy arrays for model evaluation after training
np.save(dir_anfiles+"/an_l_eval_sounds.npy",an_l_eval)
np.save(dir_anfiles+"/an_r_eval_sounds.npy",an_r_eval)
np.save(dir_anfiles+"/labels_eval_sounds.npy",labels_eval)
pickle.dump(filenames_eval, open(dir_anfiles+'/listfilenames_evall_sounds.p','wb'))

print("numpy arrays are saved to disk")
print("Shape of evaluation sounds is:", an_l_eval.shape)
print("Shape of evaluation labels is:", labels_eval.shape)


        
