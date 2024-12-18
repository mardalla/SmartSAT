from models import vae_convolutional
from configurations import ds_config
from util import dataset
import numpy as np
import os
import talos
import pickle
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
experiment_name = 'hyperparam_vae_single_'
folders = ['/projects/satdb/cnfgen_graph/dataset_preprocessed/train/']

# Data creation and load
ds = dataset.Dataset()
ds.dataset_creation(folders, limit = 5000)
ds.data_summary()

model = vae_convolutional.ConvolutionalVAE()
model.max_variables = ds.max_variables
model.max_clauses = ds.max_clauses
model.encoding_size = ds.encoding_size
model.name = experiment_name
p = {'batch_normalization': False, 'batch_size': 5, 'conv_kernel_init': 'he_normal', 'decay': 0.0001,
     'dense_activation': 'relu', 'dense_dim': 64, 'dense_dropout': 0.1, 'dense_kernel_init': 'he_normal',
     'dense_layers': 2, 'epochs': 700, 'first_conv_activation': 'relu', 'first_conv_dim': 1, 'first_conv_dropout': 0.1,
     'latent_dim': 8, 'lr': 1e-05, 'optimizer': 'adam', 'patience': 30, 'second_conv_activation': 'relu',
     'second_conv_dim': 128, 'second_conv_dropout': 0.2, 'third_conv_activation': 'selu', 'third_conv_dim': 32,
     'third_conv_dropout': 0.2, 'third_conv_layers': 3, 'third_conv_stride': 8, 'third_conv_win': 9}

model.training(ds.X_train, ds.X_train, None, None, p)


for i, t in enumerate(['train', 'test']):
    print("Starting phase ", t)
    folders = ['/projects/satdb/cnfgen_graph/dataset_preprocessed/' + t + '/']
    if i == 1:
    # Data creation and load
        ds = dataset.Dataset()
        ds.dataset_creation(folders, limit = 5000)
    ds.data_summary()
    # ds.data_save(experiment_name) Class_train_kcolor0015_mode

    print("Start encoding")
    print(model.model.summary())
    latent = model.encode(ds.X_train)
    print("Start decoding")

    pred = model.decode(latent)

    print("Original data")
    print("Mean", np.mean(ds.X_train))
    print("\nReconstruction data ")
    print("Mean", np.mean(pred))
    print("Max", np.amax(pred))
    print("\nLatent data ")
    print("\nReconstruction error ")
    print("Vae:\t ", np.mean(K.mean(K.square(ds.X_train - pred), axis=-1)))
    avg = np.mean(ds.X_train, axis=0)
    avg = np.resize(avg, ds.X_train.shape)
    print("Naive:\t ", np.mean(K.mean(K.square(ds.X_train - avg), axis=-1)))


    with open('./latent_projection/' + experiment_name + '_data_' + t + '.npy', 'wb') as f:
        np.save(f, latent)
        f.close()

    with open('./latent_projection/' + experiment_name + '_label_' + t + '.npy', 'wb') as f:
        np.save(f, ds.y_train)
        f.close()

    del ds, latent, pred

