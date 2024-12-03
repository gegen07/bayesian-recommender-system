import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

import logging
import time

import pytensor
import scipy as sp

# Enable on-the-fly graph computations, but ignore
# absence of intermediate test values.
pytensor.config.compute_test_value = "ignore"

# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


class PMF:
    """Probabilistic Matrix Factorization model using pymc."""

    def __init__(self, train, dim, alpha=2, std=0.01, bounds=(1, 5)):
        """Build the Probabilistic Matrix Factorization model using pymc.

        :param np.ndarray train: The training data to use for learning the model.
        :param int dim: Dimensionality of the model; number of latent factors.
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        :param (tuple of int) bounds: (lower, upper) bound of ratings.
            These bounds will simply be used to cap the estimates produced for R.

        """
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m = self.data.shape

        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        # Specify the model.
        logging.info("building the PMF model")
        with pm.Model(
            coords={
                "users": np.arange(n),
                "movies": np.arange(m),
                "latent_factors": np.arange(dim),
                "obs_id": np.arange(self.data[~nan_mask].shape[0]),
            }
        ) as pmf:
            U = pm.MvNormal(
                "U",
                mu=0,
                tau=self.alpha_u * np.eye(dim),
                dims=("users", "latent_factors"),
                initval=rng.standard_normal(size=(n, dim)) * std,
            )
            V = pm.MvNormal(
                "V",
                mu=0,
                tau=self.alpha_v * np.eye(dim),
                dims=("movies", "latent_factors"),
                initval=rng.standard_normal(size=(m, dim)) * std,
            )
            R = pm.Normal(
                "R",
                mu=(U @ V.T)[~nan_mask],
                tau=self.alpha,
                dims="obs_id",
                observed=self.data[~nan_mask],
            )

        logging.info("done building the PMF model")
        self.model = pmf
        
        self.model.draw_samples = self._draw_samples

        # Update our class with the new MAP infrastructure.
        self.model.find_map = self._find_map
        self.model.map = property(self._map)

    def __str__(self):
        return self.name
    
    def _find_map(self):
        """Find mode of posterior using L-BFGS-B optimization."""
        tstart = time.time()
        with self.model:
            logging.info("finding PMF MAP using L-BFGS-B optimization...")
            # start = pm.find_initial_point(vars=self.model.free_RVs)
            self._map = pm.find_MAP(method="L-BFGS-B")

        elapsed = int(time.time() - tstart)
        logging.info("found PMF MAP in %d seconds" % elapsed)
        return self._map


    def _map(self):
        try:
            return self._map
        except:
            return self.find_map()
        
    def _draw_samples(self, **kwargs):
        # kwargs.setdefault("chains", 1)
        with self.model:
            self.trace = pm.sample(**kwargs)
            
    def _predict(self, U, V):
        """Estimate R from the given values of U and V."""
        R = np.dot(U, V.T)
        sample_R = rng.normal(R, self.std)
        # bound ratings
        low, high = self.bounds
        sample_R[sample_R < low] = low
        sample_R[sample_R > high] = high
        return sample_R

def train():
    # We use a fixed precision for the likelihood.
    # This reflects uncertainty in the dot product.
    # We choose 2 in the footsteps Salakhutdinov
    # Mnihof.
    ALPHA = 2

    # The dimensionality D; the number of latent factors.
    # We can adjust this higher to try to capture more subtle
    # characteristics of each movie. However, the higher it is,
    # the more expensive our inference procedures will be.
    # Specifically, we have D(N + M) latent variables. For our
    # Movielens dataset, this means we have D(2625), so for 5
    # dimensions, we are sampling 13125 latent variables.
    DIM = 10


    pmf = PMF(train, DIM, ALPHA, std=0.05)
    pmf.find_map()
    pmf.draw_samples(draws=500, tune=500)

    import pickle 

    with open("model_output.pkl", "wb") as f:
        pickle.dump(pmf, f)
    

if __name__ == "__main__":
    train()
