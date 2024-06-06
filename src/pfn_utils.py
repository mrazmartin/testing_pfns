from torch import nn
from pfns.train import train
from pfns import encoders
from pfns import utils
from pfns import bar_distribution
from pfns.priors import Batch
from pfns.priors import get_batch_to_dataloader

from src.priors import sample_linear_prior


device = utils.default_device

def train_a_linear_pfn(get_batch_function,
                       epochs=20, max_dataset_size=20,
                       num_features=1,
                       hps=None):

    # define a bar distribution (riemann distribution) criterion with 100 bars
    ys = get_batch_function(100000,20,num_features, hyperparameters=hps).target_y.to(device)
    # we define our bar distribution adaptively with respect to the above sample of target ys from our prior
    criterion = bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_outputs=100, ys=ys))

    dataloader = get_batch_to_dataloader(get_batch_linear_regression)

    # now train
    train_result = train(# the prior is the key. It defines what we train on.
                         dataloader, criterion=criterion,
                         # define the transformer size
                         emsize=256, nhead=4, nhid=512, nlayers=4,
                         # how to encode the x and y inputs to the transformer
                         encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
                         y_encoder_generator=encoders.Linear,
                         # these are given to the prior, which needs to know how many features we have etc
                         extra_prior_kwargs_dict=\
                            {'num_features': num_features, 'fuse_x_y': False, 'hyperparameters': hps},
                         # change the number of epochs to put more compute into a training
                         # an epoch length is defined by `steps_per_epoch`
                         # the below means we do 20 epochs, with 100 batches per epoch and 8 datasets per batch
                         # that means we look at 20*100*8 = 16 000 datasets, typically we train on milllions of datasets.
                         epochs=epochs, warmup_epochs=epochs//4, steps_per_epoch=100,batch_size=8,
                         # the lr is what you want to tune! usually something in [.00005,.0001,.0003,.001] works best
                         # the lr interacts heavily with `batch_size` (smaller `batch_size` -> smaller best `lr`)
                         lr=.001,
                         # seq_len defines the size of your datasets (including the test set)
                         seq_len=max_dataset_size,
                         # single_eval_pos_gen defines where to cut off between train and test set
                         # a function that (randomly) returns lengths of the training set
                         # the below definition, will just choose the size uniformly at random up to `max_dataset_size`
                         single_eval_pos_gen=utils.get_uniform_single_eval_pos_sampler(max_dataset_size),
    )
    return train_result


def get_batch_linear_regression(batch_size=2, seq_len=100, num_features=1,
                                hyperparameters=None, bias=True, device='cpu', **kwargs):

    """
    Since we are working with the PFNs library, we need to follow their class structure.

    :param batch_size: number of datasets
    :param seq_len: number of points in each dataset
    :param num_features: number of features of each datapoint
    :param hyperparameters: dictionary with hyperparameters for the prior
    :param bias: whether to sample and add the bias term to the data
    """

    if hyperparameters is None:
        hyperparameters = {'a': 0.1, 'b': 1.0}

    xs, ys = sample_linear_prior(num_datasets=batch_size, num_features=num_features,
                               num_points_in_each_dataset=seq_len,
                               hyperparameters=hyperparameters,
                               bias=bias)

    # get_batch functions return two different ys, let's come back to this later, though.
    return Batch(x=xs.transpose(0,1).to(device), y=ys.transpose(0,1).to(device), target_y=ys.transpose(0,1).to(device))
