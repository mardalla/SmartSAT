from util import custom_keras
from keras.models import load_model
from models.model_interface import ModelInterface
import tensorflow as tf


class VaeInterface(ModelInterface):
    def __init__(self, name):
        super().__init__(name)
        self.encoder = None
        self.decoder = None
        self.train_encoder = None
        self.train_decoder = None
        self.latent_dim = 2



    def encode(self, X):
        if self.encoder == None:
            print("ERROR: the encoder needs to be trained before predict")
            return
        z, _, _ = self.encoder.predict(X, batch_size=5)
        return z

    def decode(self, X):
        if self.decoder == None:
            print("ERROR: the encoder needs to be trained before predict")
            return
        return self.decoder.predict(X, batch_size=5)

    def save_models(self):
        if self.model == None:
            print("ERROR: the model must be available before saving it")
            return
        self.model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf', save_format="tf")
        self.encoder.save(self.model_path + self.name + str(self.count_save).zfill(4)  + '_encoder.tf', save_format="tf")
        self.decoder.save(self.model_path + self.name + str(self.count_save).zfill(4)  + '_decoder.tf', save_format="tf")
        self.count_save += 1

    def load_models(self, name):
        self.model = load_model(self.model_path + name + '_model.tf', compile=False, custom_objects={'Sampling': custom_keras.Sampling})
        self.encoder = load_model(self.model_path + name + '_encoder.tf', compile=False, custom_objects={'Sampling': custom_keras.Sampling})
        print(self.encoder)
        self.decoder = load_model(self.model_path + name + '_decoder.tf', compile=False)


class VAESaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, vae):
        self.monitor = 'val_loss'
        self.vae = vae

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if logs['val_loss'] < self.vae.best_val_loss:  # your custom condition
            print('New best validation loss: ', logs['val_loss'])
            self.vae.best_val_loss = logs['val_loss']
            self.vae.model = self.vae.train_model
            self.vae.encoder = self.vae.train_encoder
            self.vae.decoder = self.vae.train_decoder
            self.vae.save_models()
            print('Model save id ', str(self.vae.count_save-1).zfill(4))
