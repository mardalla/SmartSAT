from util import dataset
import time


class DatasetConfiguration:
    def __init__(self):
        return

    def SetConfiguration(self, ds, data_path, config_name='Vib_Grundfoss'):
        if config_name == 'Vib_Grundfoss':
            ds.name = 'Vib_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 'Grundfoss'
            ds.normalizaion = 'scale'
            ds.speed_limit = 200
            ds.time_train_start = time.mktime(time.strptime("01.11.2020 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.12.2020 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        if config_name == 'Flux_Grundfoss':
            ds.name = 'Flux_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux'
            ds.machine = 'Grundfoss'
            ds.normalizaion = 'scale'
            ds.speed_limit = 200
            ds.time_train_start = time.mktime(time.strptime("01.11.2020 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.12.2020 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
