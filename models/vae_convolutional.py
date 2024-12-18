import numpy as np
import glob
from keras.layers import Input, Dense, Flatten, Reshape, Conv1D, Conv1DTranspose, BatchNormalization, Cropping1D, Dropout
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
from util import custom_keras
from util.keras_data_generator import create_generators
from models.vae_interface import VaeInterface, VAESaveCheckpoint
from models.sat_interface import SatInterface


class ConvolutionalVAE(VaeInterface, SatInterface):
    def __init__(self):
        VaeInterface.__init__(self, "ConvVAE")
        SatInterface.__init__(self)
        self.parameter_list = {'first_conv_dim': [1],
                               'first_conv_activation': ['relu'],
                               'first_conv_dropout': [0.0, 0.1, 0.2],
                               'second_conv_dim': [128, 256, 512],
                               'second_conv_activation': ['relu', 'elu', 'selu'],
                               'second_conv_dropout': [0.0, 0.1, 0.2],
                               'third_conv_layers': [1, 2, 3],
                               'third_conv_dim': [32, 128],
                               'third_conv_win': [9, 17, 33, 65],
                               'third_conv_stride': [2, 4, 8],
                               'third_conv_activation': ['relu', 'elu', 'selu'],
                               'third_conv_dropout': [0.0, 0.1, 0.2],
                               'conv_kernel_init': ['he_normal', 'glorot_uniform'],
                               'dense_layers': [1, 2],
                               'dense_dim': [32, 64, 128],
                               'dense_activation': ['relu', 'elu', 'selu'],
                               'dense_dropout': [0.0, 0.05, 0.1],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [5],
                               'epochs': [400],
                               'patience': [30],
                               'latent_dim': [8],
                               'optimizer': ['adam', 'nadam', 'rmsprop'],
                               'batch_normalization': [True, False],
                               'lr': [1E-4, 1E-5, 1E-6, 1E-7],
                               'decay': [1E-3, 1E-4, 1E-5],
                               'path' : ['/projects/satdb/satlib/dataset_satlib_preprocessed/'],
                               'train_test_split' : [0.8]
                               }


    def training(self, X_train, Y_train, X_test, Y_test, p):
        """ Encoder and Decoder creation"""

        files = glob.glob(p['path']+'*npy')
        x0 = np.load(files[0])
        x0 = np.reshape(x0,[x0.shape[0]*x0.shape[1],1])
        self.input_shape = x0.shape

        params = {'shape': self.input_shape,
                  'batch_size': self.parameter_list['batch_size'][0],
                  'n_classes': 2,
                  'shuffle': True}
        

        train_generator, validation_generator = create_generators(p['path'], p['train_test_split'], params)

        # Hyperparametrised VAE
        self.latent_dim = p['latent_dim']
        # self.input_shape = X_train.shape[1:]
        input_tensor = Input(shape=self.input_shape)
        # First con layer
        x = Conv1D(filters=p['first_conv_dim'], kernel_size=2, strides=2, input_shape=self.input_shape,
                   padding='valid', kernel_initializer=p['conv_kernel_init'],
                   activation=p['first_conv_activation'])(input_tensor)
        if p['first_conv_dropout'] > 0:
            x = Dropout(p['first_conv_dropout'])(x)

        if p['batch_normalization']:
            x = BatchNormalization()(x)
        print("Max Variables ", self.max_variables)
        x = Conv1D(filters=p['second_conv_dim'], kernel_size=self.max_variables, strides=self.max_variables,
                   padding='valid', kernel_initializer=p['conv_kernel_init'], activation=p['second_conv_activation']) (x)
        if p['second_conv_dropout'] > 0:
            x = Dropout(p['second_conv_dropout'])(x)
        if p['batch_normalization']:
            x = BatchNormalization()(x)

        for _ in range(p['third_conv_layers']):
            x = Conv1D(filters=p['third_conv_dim'], kernel_size=p['third_conv_win'], strides=p['third_conv_stride'],
                       padding='same', kernel_initializer=p['conv_kernel_init'], activation=p['third_conv_activation'])(x)
            if p['third_conv_dropout'] > 0:
                x = Dropout(p['third_conv_dropout'])(x)
            if p['batch_normalization']:
                x = BatchNormalization()(x)

        # Flattening of the tensor
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        # Dense layers
        for _ in range(p['dense_layers']):
            x = Dense(p['dense_dim'], activation=p['dense_activation'],
                      kernel_initializer=p['dense_kernel_init'])(x)
            if p['dense_dropout'] > 0:
                x = Dropout(p['dense_dropout'])(x)
            if p['batch_normalization']:
                x = BatchNormalization()(x)
        z_mean = Dense(self.latent_dim, kernel_initializer=p['dense_kernel_init'])(x)
        z_log_var = Dense(self.latent_dim, kernel_initializer=p['dense_kernel_init'])(x)


        z = custom_keras.Sampling()([z_mean, z_log_var])
        # Encoder creation
        self.train_encoder = keras.Model(inputs=[input_tensor],
                              outputs=[z_mean, z_log_var, z], name="encoder")
        # Decoder
        latent_inputs = Input(K.int_shape(z)[1:])
        # Dense layer of the latent space
        x = Dense(p['dense_dim'], activation=p['dense_activation'],
                  kernel_initializer=p['dense_kernel_init'])(latent_inputs)
        # Dense layers
        for _ in range(p['dense_layers'] - 1):
            x = Dense(p['dense_dim'], activation=p['dense_activation'],
                      kernel_initializer=p['dense_kernel_init'])(x)
            if p['batch_normalization']:
                x = BatchNormalization()(x)
        x = Dense(np.prod(shape_before_flattening[1:]),
                  activation='relu',
                  kernel_initializer=p['dense_kernel_init'])(x)
        x = BatchNormalization()(x)
        # Reverse the flattening
        x = Reshape(shape_before_flattening[1:])(x)
        # Block of conv layers with stride 2
        for _ in range(p['third_conv_layers']):
            x = Conv1DTranspose(p['third_conv_dim'], p['third_conv_win'],
                                padding='same',
                                activation=p['third_conv_activation'],
                                strides=p['third_conv_stride'],
                                kernel_initializer=p['conv_kernel_init'])(x)
            if p['batch_normalization']:
                x = BatchNormalization()(x)

        x = Conv1DTranspose(p['second_conv_dim'], self.max_variables,
                            padding='same',
                            activation=p['second_conv_activation'],
                            strides=self.max_variables,
                            kernel_initializer=p['conv_kernel_init'])(x)
        x = BatchNormalization()(x)

        output_tensor = Conv1DTranspose(1, 2, padding='same', activation="sigmoid",
                            strides=2, kernel_initializer=p['conv_kernel_init'])(x)

        shape_err = K.int_shape(output_tensor)[1] - self.input_shape[0]
        if shape_err > 0:
            print("A crop is needed of ", shape_err)
            output_tensor = Cropping1D(cropping=(0, shape_err))(output_tensor)
        self.train_decoder = keras.Model(latent_inputs, output_tensor, name="decoder")
        """ Model creation """
        _, _, z = self.train_encoder(input_tensor)
        reconstructions = self.train_decoder(z)
        self.train_model = keras.Model(inputs=[input_tensor], outputs=[reconstructions])
        self.train_encoder.summary()
        self.train_decoder.summary()
        self.train_model.summary()
        # Loss definition
        latent_loss = -0.5 * K.sum(
            1 + z_log_var - K.exp(z_log_var) - K.square(z_mean), axis=-1
        )
        # we do it to ensure the appropriate scale between it and the reconstruction loss
        scale_factor = self.input_shape[0]  # reduce it to give more important to the compactness of the latent space
        self.train_model.add_loss(K.mean(latent_loss) / scale_factor)
        # Model optimizer
        opt = None
        if p['optimizer'] == 'adam':
            opt = Adam(lr=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        # Model compilation
        self.train_model.compile(loss='mse', optimizer=opt,
                           metrics=["mean_absolute_error", "mean_absolute_percentage_error"])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])
        vae_save = VAESaveCheckpoint(self)
        # Training
        result = self.train_model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=p['epochs'],callbacks=[es, vae_save], verbose=2)
        validation_loss = np.amin(result.history['val_loss'])
        print('Best validation loss of epoch:', validation_loss)
        return result, self.train_model
