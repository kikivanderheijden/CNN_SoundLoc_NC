# script to check files in folder

# set directories
dir_anfiles = "/workspace/notebooks/sounds/train" # for DSRI
dir_wrfiles = "/workspace/notebooks/sounds_npy/train"
#dir_anfiles = r"C:\Users\kiki.vanderheijden\Documents\SoundsNeuroComputing\TRAIN" # for local testing

# import necessary packages and libraries
import os # to get info about directories
import numpy as np
import math
from pytictoc import TicToc 
t = TicToc() # create instant of class
from scipy.io import loadmat

# set parameters
time_sound = 1500 # time dimension of sound files, i.e. number of samples
nfreqs     = 99 # nr of freqs used

# start script
t.tic()
countfiles = 0
with os.scandir(dir_anfiles) as listfiles:
    for entry in listfiles:
        countfiles = countfiles+1
        #print(entry.name) # use this to display all file names
        #print(temploc) 
        #print(trainlabels)  
t.toc("scanning files took")
print("this folder contains" , countfiles, "files" )

t.tic()
# create array of location labels for the x and y coordinates. labels range from -1 to 1 in correspondence with the unit circle 
# take labels from directory for left channel but they are the same for the left and right channel
trainlabels_x = [] # initialize list of labels
trainlabels_y = []
filenames = [] # initialize list of original file names
with os.scandir(dir_anfiles) as listfiles:
    for entry in listfiles:
        # derive location from filename 
        filenames.append(entry.name[:-4]) # do not include extension
        temploc = int(entry.name[1:4]) # get azimuth location and convert to integer
        # note that the naming of the files has 0 at front, while for the unit circle 0 should be at the right, correct this first
        if temploc >= 0 and temploc <= 90:
            temploc = np.abs(temploc - 90)
        elif temploc > 90:
            temploc = np.abs(temploc - 90- 360)
        temp_xcoord = np.around(math.cos(math.radians(temploc)),3,out=None) # math.cos operates on radians so convert angle to rad first
        temp_ycoord = np.around(math.sin(math.radians(temploc)),3,out=None)
        trainlabels_x.append(temp_xcoord)
        trainlabels_y.append(temp_ycoord) 
# add together in 2D array where column 1 = x coord and column 2 = y coord
trainlabels = np.vstack((np.array(trainlabels_x),np.array(trainlabels_y)))
trainlabels = np.transpose(trainlabels)
 
t.toc("creating the labels took")
print("shape of labels is ", trainlabels.shape)    

t.tic()
fileidx = 0
countfilesdone = 0
countfiles100 = 0
# load files as list and convert to numpy array later to save time
train_an_l = [None]*trainlabels.shape[0] #intialize list
train_an_r = [None]*trainlabels.shape[0]
with os.scandir(dir_anfiles) as listfiles:
    for entry in listfiles:
        train_an_l[fileidx] = loadmat(dir_anfiles+"/"+entry.name,verify_compressed_data_integrity=False)['AN_l'] # note that indexing is faster than appending
        train_an_r[fileidx] = loadmat(dir_anfiles+"/"+entry.name,verify_compressed_data_integrity=False)['AN_r']
        countfilesdone = countfilesdone+1
        fileidx = fileidx+1
        # this is just for  debugging to keep track of the time/speed
        if countfilesdone == 100:
            print(countfilesdone*countfiles100, " files done")
            countfiles100 = countfiles100+1
            t.toc("these 100 files took")
            t.toc(restart=True)
            countfilesdone = 0
t.toc("loading the sounds took ")
print("shape of sounds is", len(train_an_l))
# convert to nunmpy array
train_an_l_array = np.asarray(train_an_l)
train_an_r_array = np.asarray(train_an_r)
print("shape of the sound array is ", train_an_l_array.shape)

# save numpy arrays and file names
np.save(dir_wrfiles+"/an_l_sounds.npy",train_an_l_array)
np.save(dir_wrfiles+"/an_r_sounds.npy",train_an_r_array)
np.save(dir_wrfiles+"/labels_sounds.npy",trainlabels)

import pickle
pickle.dump(filenames, open(dir_wrfiles+'/listfilenames_sounds.p','wb'))
