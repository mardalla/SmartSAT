from models import vae_convolutional, pca
from configurations import ds_config
from util import dataset, plotter
import time
import numpy as np

data_path = '/Users/andreavisentin/ADI/data_tm/'

# Data creation and load
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path,'Flux_Grundfoss')
# ds.dataset_creation()
# ds.data_save(ds.name)
ds = ds.data_load(ds.name)
ds.data_summary()
pca_model = pca.PCAModel()

pca_model.training(ds.X_train)

x_latent = pca_model.encode(ds.X_test)
print(x_latent.shape)
y = pca_model.decode(x_latent)
print(y.shape)


# vae = convolutional_vae.ConvolutionalVAE()
# err_time = time.mktime(time.strptime("10.02.2021 09:40:00", "%d.%m.%Y %H:%M:%S"))
# metadata_fault = ds.metadata_test[:,2] > err_time
# metadata_fault[np.where(ds.metadata_test[:,1]<1000)] =2
# print(sum(ds.metadata_test[:,1]<1000))
# vae.load_models("Flux_Grundfoss0047")
#
# plotter.plot_latent_space_grid_4(vae, ds.X_test, metadata_fault, [0,2,4,6], [1,3,5,7])