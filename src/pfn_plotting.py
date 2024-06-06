from matplotlib import pyplot as plt

def plot_datasets(xs, ys):
    n_datasets = xs.shape[0]
    for dataset_index in range(n_datasets):
        plt.scatter(xs[dataset_index,:,0].numpy(), ys[dataset_index].numpy())
    plt.show()