import torch

def sample_linear_prior(num_datasets = 10, num_features=1, num_points_in_each_dataset = 100,
                      hyperparameters={'a': 0.1, 'b': 1.0}, bias=True):
    """
    This is an example of a simple prior. Based on this code, you will later implement a more complex prior.

    Let's assume we have a simple linear mapping:
        f = w*x + bias     with   y ~ N(f, a^2)

    we define our prior over our 'latent' -> w, including bias, as a normal distribution with mean 0 and variance b^2:
        w ~ N(0, ^2)
    """    

    ws = torch.distributions.Normal(torch.zeros(num_features+1), hyperparameters['b']).sample((num_datasets,))

    # sample the xs from a uniform distribution between 0 and 1
    xs = torch.rand(num_datasets, num_points_in_each_dataset, num_features)
    
    if bias:
        # add a bias term so our lines don't start at the origin
        xs = torch.cat([xs,torch.ones(num_datasets, num_points_in_each_dataset,1)], 2)

    ys = torch.distributions.Normal(
        torch.einsum('nmf, nf -> nm',
                     xs,
                     ws
                    ),
        hyperparameters['a']
    ).sample()

    return xs, ys

def sample_cos_prior(num_datasets=10, num_features=1, num_points_in_each_dataset=100,
                        hyperparameters={'a': 0.1, 'b': 1.0}, bias=True):
    ...
