from models import classifier_cnn
from configurations import ds_config
from util import dataset, plotter
import os



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
experiment_name = 'Model_tester'

folders = ['/projects/satdb/cnfgen_graph/dataset_kcolor_preprocessed/train/']
# Data creation and load
ds = dataset.Dataset()
ds.dataset_creation(folders, limit = 200)
ds.data_summary()
# ds.data_save(experiment_name)

model = classifier_cnn.ClassifierCNN()
model.name = experiment_name
model.max_variables = ds.max_variables
model.max_clauses = ds.max_clauses
model.encoding_size = ds.encoding_size
model.verbose = True

p ={'batch_normalization': False, 'batch_size': 5, 'conv_kernel_init': 'he_normal', 'decay': 1e-05, 'dense_activation': 'elu', 'dense_dim': 64, 'dense_dropout': 0.1, 'dense_kernel_init': 'glorot_uniform', 'dense_layers': 2, 'epochs': 400, 'first_conv_activation': 'relu', 'first_conv_dim': 1, 'first_conv_dropout': 0.2, 'lr': 1e-06, 'optimizer': 'rmsprop', 'patience': 30, 'second_conv_activation': 'elu', 'second_conv_dim': 128, 'second_conv_dropout': 0.1, 'third_conv_activation': 'elu', 'third_conv_dim': 32, 'third_conv_dropout': 0.2, 'third_conv_layers': 4, 'third_conv_stride': 9, 'third_conv_win': 9}
model.training(ds.X_train, ds.y_train, None, None, p)

