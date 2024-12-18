import numpy as np
import glob
import keras
from keras.layers import Dense
from keras.layers import Conv1D, Flatten, Activation, Dropout, BatchNormalization, Input
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import EarlyStopping
from util import custom_keras
from util.keras_data_generator import create_generators
from models.sat_interface import SatInterface
from models.model_interface import ModelInterface


class ClassifierCNN(ModelInterface, SatInterface):
    def __init__(self):
        ModelInterface.__init__(self, "ClassifierCNN")
        SatInterface.__init__(self)
        self.parameter_list = {'first_conv_dim': [1],
                               'first_conv_activation': ['relu'],
                               'first_conv_dropout': [0.0, 0.1, 0.2],
                               'second_conv_dim': [128, 256, 512],
                               'second_conv_activation': ['relu', 'elu', 'selu'],
                               'second_conv_dropout': [0.0, 0.1, 0.2],
                               'third_conv_layers': [0, 2, 4, 6],
                               'third_conv_dim': [32, 128, 512],
                               'third_conv_win': [9, 17, 33, 65],
                               'third_conv_stride': [1, 9],
                               'third_conv_activation': ['relu', 'elu', 'selu'],
                               'third_conv_dropout': [0.0, 0.1, 0.2],
                               'conv_kernel_init': ['he_normal', 'glorot_uniform'],
                               'dense_layers': [1, 2, 3],
                               'dense_dim': [32, 64, 128, 256],
                               'dense_activation': ['relu', 'elu', 'selu'],
                               'dense_dropout': [0.0, 0.05, 0.1],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [5],
                               'epochs': [400],
                               'patience': [30],
                               'optimizer': ['adam', 'nadam', 'rmsprop'],
                               'batch_normalization': [True, False],
                               'lr': [1E-2, 1E-3, 1E-4, 1E-5, 1E-6],
                               'decay': [1E-3, 1E-4, 1E-5],
                               'path' : ['/projects/satdb/satlib/dataset_satlib_preprocessed/'],
                               'train_test_split' : [0.8]
                               }

    def training(self, X_train, y_train, X_test, y_test, p):
        # self.input_shape = X_train.shape[1:]

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
        # First con layer
        x = Conv1D(filters=p['first_conv_dim'], kernel_size=2, strides=2, input_shape=self.input_shape,
                   padding='valid', kernel_initializer=p['conv_kernel_init'],
                   activation=p['first_conv_activation'])(input_tensor)
        if p['first_conv_dropout'] > 0:
            x = Dropout(p['first_conv_dropout'])(x)

        if p['batch_normalization']:
            x = BatchNormalization()(x)
        x = Conv1D(filters=p['second_conv_dim'], kernel_size=self.max_variables, strides=self.max_variables,
                   padding='valid', kernel_initializer=p['conv_kernel_init'], activation=p['second_conv_activation'])(x)
        if p['second_conv_dropout'] > 0:
            x = Dropout(p['second_conv_dropout'])(x)
        if p['batch_normalization']:
            x = BatchNormalization()(x)

        for _ in range(p['third_conv_layers']):
            x = Conv1D(filters=p['third_conv_dim'], kernel_size=p['third_conv_win'], strides=p['third_conv_stride'],
                       padding='same', kernel_initializer=p['conv_kernel_init'], activation=p['third_conv_activation'])(
                x)
            if p['third_conv_dropout'] > 0:
                x = Dropout(p['third_conv_dropout'])(x)
            if p['batch_normalization']:
                x = BatchNormalization()(x)

        # Flattening of the tensor
        x = Flatten()(x)
        # Dense layers
        for _ in range(p['dense_layers']):
            x = Dense(p['dense_dim'], activation=p['dense_activation'],
                      kernel_initializer=p['dense_kernel_init'])(x)
            if p['dense_dropout'] > 0:
                x = Dropout(p['dense_dropout'])(x)
            if p['batch_normalization']:
                x = BatchNormalization()(x)
        output_tensor = Dense(1, activation="sigmoid")(x)
        self.train_model = keras.Model(inputs=input_tensor, outputs=output_tensor)

        if self.verbose:
            self.train_model.summary()
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
        save_check = custom_keras.CustomSaveCheckpoint(self)
        # Training
        result = self.train_model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=p['epochs'], callbacks=[es, save_check], verbose=2)
        validation_loss = np.amin(result.history['val_loss'])
        print('Best validation loss of epoch:', validation_loss)
        return result, self.train_model
