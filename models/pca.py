from sklearn.decomposition import PCA
import numpy as np
from models.model_interface import ModelInterface

class PCAModel(ModelInterface):
    def __init__(self):
        ModelInterface.__init__(self, "PCA")
        self.pca = None
        self.channels = 0

    def predict(self, X):
        if self.pca is None:
            print("ERROR: the model needs to be trained before predict")
            return
        return self.decode(self.encode(X))

    def encode(self, X):
        if self.pca is None:
            print("ERROR: the encoder needs to be trained before predict")
            return
        out = np.zeros((X.shape[0], self.latent_dim, self.channels))
        for i in range(self.channels):
            out[:,:,i] = self.pca[i].transform(X[:,:,i])
        return out

    def decode(self, X):
        if self.pca is None:
            print("ERROR: the encoder needs to be trained before predict")
            return
        out = np.zeros((X.shape[0], 15000, self.channels))
        for i in range(self.channels):
            out[:,:,i] = self.pca[i].inverse_transform(X[:,:,i])
        return out


    def training(self, X_train, Y_train = None, X_test = None, Y_test = None, p = None):
        self.pca = {}
        self.channels = X_train.shape[2]
        for i in range(self.channels):
            self.pca[i] = PCA(n_components=self.latent_dim)
            self.pca[i].fit(X_train[:,:,i])

