#!/usr/bin/env python
# coding: utf-8

import numpy as np
import keras
import glob
import random


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=16, shape=(960*450, 1), n_classes=2, shuffle=True):
        'Initialization'
        self.shape = shape
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.shape))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            array = np.load(ID)
            array = np.reshape(array, [*self.shape])
            X[i,] = array
            # Store class
            y[i] = self.labels[ID]

        if self.n_classes > 2 :
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

def create_generators(path='../Datasets/dataset_normalized/', train_size=0.8, params=None):

    if params == None:
        default_parameters = {'shape': (960*450,1),
            'batch_size': 16,
            'n_classes': 2,
            'shuffle': True}
        params = default_parameters

    # Separate the dataset in train & validation
    list_of_files = glob.glob(path+'*npy')
    random.shuffle(list_of_files)
    train = list_of_files[:int(len(list_of_files) * train_size)]
    validation = list_of_files[int(len(list_of_files) * train_size):]

    # Create dictionaries for data and labels

    partition = {'train' : train, 'validation' : validation}

    labels = {}

    for file in list_of_files:
        name = file.replace(path,'')
        if name[0] == 's':
            labels[file] = 1
        elif name[0] == 'u':
            labels[file] = 0


    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    return training_generator, validation_generator




