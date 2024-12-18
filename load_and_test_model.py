from models import classifier_cnn, vae_convolutional
from configurations import ds_config
from util import dataset
import os
import talos
import pickle
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
experiment_name = 'Test_experiment'

folders = ['/projects/satdb/cnfgen_graph/dataset_preprocessed/test/']
# Data creation and load
ds = dataset.Dataset()
ds.dataset_creation(folders)
ds.data_summary()
# ds.data_save(experiment_name) Class_train_kcolor0015_mode

model = classifier_cnn.ClassifierCNN()
model.max_variables = ds.max_variables
model.max_clauses = ds.max_clauses
model.encoding_size = ds.encoding_size
model.name = experiment_name
model.load_model("Class_train_kcolor0025")
model.model.evaluate(ds.X_train, ds.y_train)

# data = ds.X_train[0:10,:,:]
# pred = model.predict(ds.X_train[0:10,:,:])
#
# print(np.mean(data))
#
#
# print(np.mean(pred))
# print(np.amax(pred))
#
# print(pred)