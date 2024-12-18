import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import random
import math
import glob
from models.sat_interface import SatInterface

class Dataset(SatInterface):
    def __init__(self):
        SatInterface.__init__(self)
        # Definition of all the instance attributes
        # Name of the experiment
        self.name = "Dataset"
        # Training instances
        self.X_train = []
        # Test instances
        self.X_test = []
        # Training labels
        self.y_train = []
        # Test labels
        self.y_test = []
        # Training metadata
        self.metadata_train = []
        # Test metadata
        self.metadata_test = []

        # Input files
        self.cnf_location = './cnfgen_graph/dataset_kcolor_preprocessed/'
        self.metadata_file = None
        self.data_path = './saved_data/'
        self.verbose = 1


    def data_save(self, name):
        with open(self.data_path + name, 'wb') as file:
            # Step 3
            pickle.dump(self, file)
            print("File saved in " + self.data_path + name)

    def data_load(self, name):
        with open(self.data_path + name, 'rb') as file:
            # Step 3
            return pickle.load(file)

    def data_summary(self):
        print('Training set', self.X_train.shape, 'Training lables', self.y_train.shape)

    def dataset_creation(self, folders = None, limit = 5000):
        if self.verbose:
            print("Data load")

        files = []
        if folders is None:
            files = glob.glob(self.cnf_location + '*npy')
        else:
            for folder in folders:
                files += glob.glob(folder + '*npy')
        if len(files) > limit:
            files = random.sample(files, limit)
        print(len(files))
        array = np.load(files[0])
        print(array.shape)
        self.max_clauses = array.shape[0]
        self.max_variables = array.shape[1] // 2
        self.encoding_size = 2 * self.max_variables * self.max_clauses
        X = np.zeros((len(files), self.encoding_size, 1))

        for i, f in enumerate(files):
            array = np.load(f)
            X[i,:,0] = array.flatten()
        y = []
        for f in files:
            if '/sat_' in f:
                y.append(1.)
            elif '/uf' in f:
                y.append(1.)
            elif '/uuf' in f:
                y.append(0.)
            elif '/unsat_' in f:
                y.append(0.)
        print(X.shape)
        y = np.array(y, dtype='float32')
        print(np.mean(y))
        self.X_train = X
        self.y_train = y