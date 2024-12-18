from models import classifier_cnn
from configurations import ds_config
from util import dataset
import os
import talos
import pickle


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
experiment_name = 'satlib_conv'

folders = ['/projects/satdb/dataset_satlib_preprocessed/']
# Data creation and load
ds = dataset.Dataset()
ds.dataset_creation(folders)
ds.data_summary()
# ds.data_save(experiment_name)

model = classifier_cnn.ClassifierCNN()
model.max_variables = ds.max_variables
model.max_clauses = ds.max_clauses
model.encoding_size = ds.encoding_size
model.name = experiment_name
t = talos.Scan(x=ds.X_train,
               y=ds.y_train,
               model=model.training,
               experiment_name=experiment_name,
               params=model.parameter_list,
               round_limit=100,
               print_params=True)

filehandler = open("./talos_results/" + ds.name + ".obj", 'wb')
pickle.dump(t.data, filehandler)
