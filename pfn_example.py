
try:
    from PFNs import pfns
except ImportError:
    raise ImportError("Please restart runtime by i) clicking on \'Runtime\' and then ii) clicking \'Restart runtime\'")

import matplotlib.pyplot as plt

from pfns import utils

from src.pfn_plotting import plot_datasets
from src.priors import sample_linear_prior
from src.pfn_utils import train_a_linear_pfn, get_batch_linear_regression

device = utils.default_device

"""
In this exercise, we will try to showcase and play around with the PFNs library.

1. We will sample some data from a prior distribution.
    a. We will sample data (priors) with and without a bias term, plotting the results.
    b. We will improve our priors by adding a bias term.

2. We will prepare the data for our transformer model.
"""

SHOW_PLOTS = True # feel free to set to False to speed up the execution

if __name__ == '__main__':

    xs, ys, = sample_linear_prior(num_datasets=10, num_features=1,
                      num_points_in_each_dataset=100,
                      hyperparameters={'a': 0.1, 'b': 1.0},
                      bias=False)
    
    if SHOW_PLOTS:
        plot_datasets(xs, ys)

    """
    hmm, those are ok, but we can do better.
    """

    xs, ys = sample_linear_prior(num_datasets=10,
                               num_features=1,
                               num_points_in_each_dataset=100,
                               hyperparameters={'a': 0.1, 'b': 1.0},
                               bias=True,)
    
    if SHOW_PLOTS:
        plot_datasets(xs, ys)

    """
    NICE! We have a rough idea about how our data looks like.
    Now is time to prepare the data for our transformer model.
    
    We will follow a different naming convention for the variables:
        num_datasets -> batch_size (we treat a dataset as a single input point)
        num_points_in_each_dataset -> seq_len

    As we are working with the pfn library, we will need to follow their class structure.
    """

    small_batch = get_batch_linear_regression(batch_size=3, seq_len=13, num_features=1,)
    print(f"test_batch.x.shape: {small_batch.x.shape},\n"
          f"test_batch.y.shape: {small_batch.y.shape},\n"
          f"test_batch.target_y.shape: {small_batch.target_y.shape}")
    # notice that number of features for small_batch.x is (num_features + 1) because we added the bias term

    """
    Now the very last step, let's train our model!
    """
    train_result = train_a_linear_pfn(get_batch_linear_regression)
    
