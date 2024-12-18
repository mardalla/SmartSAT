import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean, log_var = inputs
        return mean + K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2)


class CustomSaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, vae):
        self.monitor = 'val_loss'
        self.vae = vae

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if logs['val_loss'] < self.vae.best_val_loss:  # your custom condition
            print('New best validation loss: ', logs['val_loss'])
            self.vae.best_val_loss = logs['val_loss']
            self.vae.model = self.vae.train_model
            self.vae.save_model()
            print('Model save id ', str(self.vae.count_save-1).zfill(4))



