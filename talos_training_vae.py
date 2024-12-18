from models import vae_convolutional
from configurations import ds_config
from util import dataset
import os
import talos
import pickle


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
experiment_name = 'hyperparam_vae'
folders = ['/projects/satdb/cnfgen_graph/dataset_preprocessed/train/']

# Data creation and load
ds = dataset.Dataset()
ds.dataset_creation(folders, limit = 3000)
ds.data_summary()
# ds.data_save(experiment_name)

model = vae_convolutional.ConvolutionalVAE()
model.max_variables = ds.max_variables
model.max_clauses = ds.max_clauses
model.encoding_size = ds.encoding_size
model.name = experiment_name
t = talos.Scan(x=ds.X_train,
               y=ds.y_train,
               model=model.training,
               experiment_name=experiment_name,
               params=model.parameter_list,
               round_limit=50,
               print_params=True)

filehandler = open("./talos_results/" + ds.name + ".obj", 'wb')
pickle.dump(t.data, filehandler)
