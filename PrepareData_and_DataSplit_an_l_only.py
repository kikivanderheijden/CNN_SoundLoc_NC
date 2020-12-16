
# read and preprocess sounds in the DSRI environment

# define directories, add r before the name because a normal string cannot be used as a path, alternatives are 
# using / or \\ instead of \
dir_anfiles = "/workspace/notebooks/sounds_npy/train" # directory for both channels
#dir_anfiles = "C:/Users/kiki.vanderheijden/Documents/PYTHON/DataFiles" # sounds left channel

# import packages and libraries
import numpy as np
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pytictoc import TicToc 
t = TicToc() # create instant of class
import pickle
import math

# set parameters
testset = 0.25 # size of testset
nrazlocs = 36 # number of azimuth locations
azimuthrange = np.arange(0,360,10)

t.tic()
# =============================================================================
# THESE ARE THE REAL ONES
# load numpy arrays from disk
an_l = np.load(dir_anfiles+"/an_l_sounds.npy")
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
an_l_train = np.zeros((nrazlocs*math.floor((1-testset)*nrsounds_loc),len(an_l[1,:,:]),len(an_l[1,1,:]))) # round down here because train_test_split rounds the train size down
labels_train = np.zeros((nrazlocs*math.floor((1-testset)*nrsounds_loc),len(labels[1,:])))
filenames_train = list()
an_l_test = np.zeros((nrazlocs*math.ceil(testset*nrsounds_loc),len(an_l[1,:,:]),len(an_l[1,1,:]))) # round up here because train_test_split rounds the test size up
labels_test = np.zeros((nrazlocs*math.ceil(testset*nrsounds_loc),len(labels[1,:])))
filenames_test = list()

# first take a part away for evaluation
for x in range(nrazlocs) :
    #  now take all those sounds --> you don't need to shuffle them because os.scandir already read them in an arbitrary order
    temp_idx = indices_val_angle[x,]
    temp_idx = temp_idx.astype(int)
    an_l_temp = an_l[temp_idx,]
    labels_temp = labels[temp_idx,]
    filenames_temp = [filenames[i] for i in temp_idx] # indexing from a list is a bit cumbersome...
    # now you need to split the remaining ones into training and test
    an_l_temp_train, an_l_temp_test, labels_temp_train, labels_temp_test, filenames_temp_train, filenames_temp_test = train_test_split(an_l_temp, labels_temp, filenames_temp, test_size = testset, shuffle = False) 
    # add to the matrices in the correct positions
    an_l_train[x*len(an_l_temp_train):(x+1)*len(an_l_temp_train)] = an_l_temp_train
    an_l_test[x*len(an_l_temp_test):(x+1)*len(an_l_temp_test)] = an_l_temp_test
    labels_train[x*len(labels_temp_train):(x+1)*len(labels_temp_train)] = labels_temp_train
    labels_test[x*len(labels_temp_test):(x+1)*len(labels_temp_test)] = labels_temp_test
    filenames_train.extend(filenames_temp_train)
    filenames_test.extend(filenames_temp_test)

# add a fourth dimension ('channel') to train_an_l and train_an_r which should be 1, this is needed for the input to the DNN
#an_l_train = np.expand_dims(an_l_train,axis = 3)
#an_l_test = np.expand_dims(an_l_test,axis = 3)


#save numpy arrays for model evaluation after training
np.save(dir_anfiles+"/an_l_train_sounds.npy",an_l_train)
np.save(dir_anfiles+"/an_l_test_sounds.npy",an_l_test)
np.save(dir_anfiles+"/labels_train_sounds.npy",labels_train)
np.save(dir_anfiles+"/labels_test_sounds.npy",labels_test)
pickle.dump(filenames_train, open(dir_anfiles+'/listfilenames_train_sounds.p','wb'))
pickle.dump(filenames_test, open(dir_anfiles+'/listfilenames_test_sounds.p','wb'))

print("numpy arrays are saved to disk")
print("Shape of training sounds is:", an_l_train.shape)
print("Shape of training labels is:", labels_train.shape)
print("Shape of test sounds is:", an_l_test.shape)
print("Shape of test labels is:", labels_test.shape)

        
