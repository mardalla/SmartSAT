from models import vae_convolutional

import glob
import os
import talos
import pickle

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


folder = '/projects/satdb/cnfgen_graph/dataset_preprocessed/train/'
files = glob.glob(folder + '*npy')
x0 = np.load(files[0])
model = vae_convolutional.ConvolutionalVAE()
experiment_name = 'VAE_convolutional_generator'
model.name = experiment_name
model.parameter_list['path'] = [folder]
model.max_variables = int(x0.shape[1]/2)
model.max_clauses = x0.shape[0]
model.encoding_size = x0.shape[0]*x0.shape[1]
model.verbose = True
dummy_x = np.empty((1,model.max_variables*2, model.max_clauses, 1))
dummy_y = np.empty((1, 1))
testX = np.empty((1,model.max_variables*2, model.max_clauses, 1))
testY = np.empty((1, 1))



t = talos.Scan(x=dummy_x,
               y=dummy_y,
               x_val=testX,
               y_val=testY,
               model=model.training,
               experiment_name=experiment_name,
               params=model.parameter_list,
               round_limit=50)

filehandler = open("./" + ds.name + ".obj", 'wb')
pickle.dump(t.data, filehandler)