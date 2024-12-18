import matplotlib.pyplot as plt
import numpy as np

def plot_latent_space(model, data, labels, index_x=0, index_y=1):
    # display a 2D plot of the digit classes in the latent space
    z = model.encode(data)
    plt.figure(figsize=(12, 10))
    if labels is None:
        plt.scatter(z[:, index_x], z[:, index_y])
    else:
        plt.scatter(z[:, index_x], z[:, index_y], c=labels)
    plt.colorbar()
    plt.xlabel("z[" + str(index_x) + "]")
    plt.ylabel("z[" + str(index_y) + "]")
    plt.margins(0)
    plt.show()

def plot_latent_space_grid_4(model, data, labels, xs, ys):
    # display a 2D plot of the digit classes in the latent space
    plt.figure(figsize=(24, 20))

    z = model.encode(data)
    print(z.shape)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Latent space plotting')
    ax1.scatter(z[:, xs[0]], z[:, ys[0]], c=labels)
    ax2.scatter(z[:, xs[1]], z[:, ys[1]], c=labels)
    ax3.scatter(z[:, xs[2]], z[:, ys[2]], c=labels)
    ax4.scatter(z[:, xs[3]], z[:, ys[3]], c=labels)

    for ax in fig.get_axes():
        ax.label_outer()
    plt.show()

def reconstruction_error_comparison(solvers, X, limit):
    nr_solvers = len(solvers)
    nr_samples = X.shape[1]
    errors = np.zeros(nr_solvers)
