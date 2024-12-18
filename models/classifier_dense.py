import numpy as np
from keras.layers import Dense, Input
from keras.layers import Dropout, BatchNormalization, Lambda, Concatenate
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import EarlyStopping
import keras
from util import custom_keras
from models.sat_interface import SatInterface
from models.model_interface import ModelInterface


class ClassifierDense(ModelInterface, SatInterface):
    def __init__(self):
        ModelInterface.__init__(self, "ClassifierDense")
        SatInterface.__init__(self)
        self.parameter_list = {'variable_dense_activation': ['relu', 'elu', 'selu'],
                               'clauses_dense_dim': [64, 128, 256, 512],
                               'clauses_dense_activation': ['relu', 'elu', 'selu'],
                               'clauses_dense_dropout': [0.0, 0.05, 0.1],
                               'dense_layers': [1, 2, 3, 4, 5],
                               'dense_dim': [32, 64, 128, 256],
                               'dense_activation': ['relu', 'elu', 'selu'],
                               'dense_dropout': [0.0, 0.05, 0.1],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [20],
                               'epochs': [400],
                               'patience': [20],
                               'optimizer': ['adam', 'nadam', 'rmsprop'],
                               'batch_normalization': [True, False],
                               'lr': [1E-2, 1E-3, 1E-4, 1E-5],
                               'decay': [1E-2, 1E-3, 1E-4, 1E-5]
                               }

    def training(self, X_train, y_train, X_test, y_test, p):
        """ Encoder and Decoder creation"""
        input_tensor = Input(shape=self.input_shape)

        # First layer at variable level
        variable_dense = Dense(1, activation=p['variable_dense_activation'], kernel_initializer=p['dense_kernel_init'])
        var_outputs = []
        for i in range(self.max_variables*self.max_clauses):
            out = Lambda(lambda x: x[2 * i:2 * (i + 1), :])(input_tensor)

            # Setting up your per-channel layers (replace with actual sub-models):
            out = variable_dense(out)
            var_outputs.append(out)
        clauses_output =[]
        clauses_dense = Dense(p['clauses_dense_dim'], activation=p['clauses_dense_activation'], kernel_initializer=p['dense_kernel_init'])
        print(len(var_outputs))
        for i in range(self.max_clauses):
            out = Concatenate()(var_outputs[i*self.max_variables: (i+1)*self.max_variables])

            # Setting up your per-channel layers (replace with actual sub-models):
            out = clauses_dense(out)
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
        # Training
        result = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                                      validation_split=0.2,
                                      callbacks=[es, vae_save], verbose=2)
        validation_loss = np.amin(result.history['val_loss'])
        print('Best validation loss of epoch:', validation_loss)
        return result, self.train_model
