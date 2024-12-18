import numpy as np
import glob
from keras.models import load_model

from keras.layers import Dense, Input
from keras.layers import Conv1D, Flatten, Activation, Dropout, BatchNormalization, Lambda, Concatenate
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
from util import custom_keras
from util.keras_data_generator import create_generators
from models.sat_interface import SatInterface
from models.model_interface import ModelInterface


class ClassifierConvDense(ModelInterface, SatInterface):
    def __init__(self):
        ModelInterface.__init__(self, "ClassifierConvDense")
        SatInterface.__init__(self)
        self.parameter_list = {'first_conv_dim': [1],
                               'first_conv_activation': ['relu', 'elu', 'selu'],
                               'first_conv_dropout': [0.0, 0.05, 0.1],
                               'conv_kernel_init': ['he_normal', 'glorot_uniform'],
                               'clauses_dense_dim': [32, 64],
                               'clauses_dense_activation': ['relu', 'elu', 'selu'],
                               'clauses_dense_dropout': [0.0, 0.05, 0.1],
                               'dense_layers': [1, 2, 3, 4],
                               'dense_dim': [32, 64, 128],
                               'dense_activation': ['relu', 'elu', 'selu'],
                               'dense_dropout': [0.0, 0.05, 0.1],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [5],
                               'epochs': [400],
                               'patience': [20],
                               'optimizer': ['adam', 'nadam', 'rmsprop'],
                               'batch_normalization': [True, False],
                               'lr': [1E-2, 1E-3, 1E-4, 1E-5],
                               'decay': [1E-2, 1E-3, 1E-4, 1E-5],
                               'path' : ['/projects/satdb/satlib/dataset_satlib_preprocessed/'],
                               'train_test_split' : [0.8]
                               }

    def training(self, X_train, y_train, X_test, y_test, p):
        """ Encoder and Decoder creation"""
        # input_tensor = Input(shape=(self.encoding_size,1))

        files = glob.glob(p['path']+'*npy')
        x0 = np.load(files[0])
        x0 = np.reshape(x0,[x0.shape[0]*x0.shape[1],1])
        self.input_shape = x0.shape
        
        params = {'shape': self.input_shape,
                  'batch_size': self.parameter_list['batch_size'][0],
                  'n_classes': 2,
                  'shuffle': True}
        

        train_generator, validation_generator = create_generators(p['path'], p['train_test_split'], params)

        input_tensor = Input(shape=self.input_shape)

        # First layer at variable level
        x = Conv1D(filters=p['first_conv_dim'], kernel_size=2, strides=2, input_shape=self.input_shape,
                   padding='valid', kernel_initializer=p['conv_kernel_init'])(input_tensor)

        if p['first_conv_dropout'] > 0:
            x = Dropout(p['first_conv_dropout'])(x)
        if p['batch_normalization']:
            x = BatchNormalization()(x)
        # print(x.shape)

        clauses_output = []
        clauses_dense = Dense(p['clauses_dense_dim'], activation=p['clauses_dense_activation'],
                              kernel_initializer=p['dense_kernel_init'])
        for i in range(self.max_clauses):
            out = clauses_dense(x[:,i * self.max_variables: (i + 1) * self.max_variables,0])

            # print(out.shape)
            clauses_output.append(out)
        x = Concatenate()(clauses_output)
        if p['clauses_dense_dropout'] > 0:
            x = Dropout(p['clauses_dense_dropout'])(x)
        if p['batch_normalization']:
            x = BatchNormalization()(x)
        for _ in range(p['dense_layers']):
            x = Dense(p['dense_dim'], activation=p['dense_activation'], kernel_initializer=p['dense_kernel_init'])(x)
            if p['dense_dropout'] > 0:
                x = Dropout(p['clauses_dense_dropout'])(x)
            if p['batch_normalization']:
                x = BatchNormalization()(x)
        x = Dense(1, activation='sigmoid')(x)

        self.train_model = keras.Model(inputs=[input_tensor], outputs=[x])
        opt = None
        if p['optimizer'] == 'adam':
            opt = Adam(lr=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        # Model compilation
        self.train_model.compile(loss='binary_crossentropy', optimizer=opt,
                                 metrics=["accuracy"])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])
        vae_save = custom_keras.CustomSaveCheckpoint(self)
        # self.train_model.summary()
        # Training
        result = self.train_model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=p['epochs'], callbacks=[es, vae_save], verbose=2)
        validation_loss = np.amin(result.history['val_loss'])
        print('Best validation loss of epoch:', validation_loss)
        return result, self.train_model
