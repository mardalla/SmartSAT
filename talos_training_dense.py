from models import classifier_cnn, classifier_conv_dense
from configurations import ds_config
from util import dataset
import os
import talos
import pickle


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
experiment_name = 'Test_dense'
folders = ['/projects/satdb/cnfgen_graph/dataset_preprocessed/train/']

# Data creation and load
ds = dataset.Dataset()
ds.dataset_creation(folders, limit = 1000)
ds.data_summary()
# ds.data_save(experiment_name)

model = classifier_conv_dense.ClassifierConvDense()
model.max_variables = ds.max_variables
model.max_clauses = ds.max_clauses
model.encoding_size = ds.encoding_size
model.name = experiment_name
print(model.max_variables)
print(model.max_clauses)
print(model.encoding_size)
print(ds.X_train.shape)
t = talos.Scan(x=ds.X_train,
               y=ds.y_train,
               model=model.training,
               experiment_name=experiment_name,
               params=model.parameter_list,
               round_limit=50,
               print_params=True)

filehandler = open("./talos_results/" + ds.name + ".obj", 'wb')
pickle.dump(t.data, filehandler)
