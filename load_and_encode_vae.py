from models import classifier_cnn, vae_convolutional
from configurations import ds_config
from util import dataset
import os
import talos
import pickle
import numpy as np
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
experiment_name = 'VAE_test'
model = vae_convolutional.ConvolutionalVAE()

for i, t in enumerate(['train', 'test']):
    print("Starting phase ", t)
    folders = ['/projects/satdb/cnfgen_graph/dataset_preprocessed/' + t + '/']
    # Data creation and load
    ds = dataset.Dataset()
    ds.dataset_creation(folders, limit = 3000)
    ds.data_summary()
    # ds.data_save(experiment_name) Class_train_kcolor0015_mode
    if i == 0:
        model.max_variables = ds.max_variables
        model.max_clauses = ds.max_clauses
        model.encoding_size = ds.encoding_size
        model.name = experiment_name
        model.load_model("v")

    pred = model.predict(ds.X_train)
    latent = model.encode(ds.X_train)

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

    del ds, pred, latent
