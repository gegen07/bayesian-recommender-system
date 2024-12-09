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

        self.data = pd.read_pickle(train).copy()
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
        
        self.draw_samples = self._draw_samples
        self.find_map = self._find_map
        self.map = property(self._map)
        self.eval_map = self.eval_map

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
    
    def rmse(self, test_data, predicted):
        """Calculate root mean squared error.
        Ignoring missing values in the test data.
        """
        I = ~np.isnan(test_data)  # indicator for missing values
        N = I.sum()  # number of non-missing values
        sqerror = abs(test_data - predicted) ** 2  # squared error array

        mse = sqerror[I].sum() / N  # mean squared error
        return np.sqrt(mse)  # RMSE
    
    def eval_map(self, train, test, k=5):
        U = self._map["U"]
        V = self._map["V"]
        # Make predictions and calculate RMSE on train & test sets.
        predictions = self._predict(U, V)
        train_rmse = self.rmse(train, predictions)
        test_rmse = self.rmse(test, predictions)

        train_avg_ndcg = self.compute_ndcg_for_all_users(predictions, train, k)
        test_avg_ndcg = self.compute_ndcg_for_all_users(predictions, test, k)

        return train_rmse, test_rmse, train_avg_ndcg, test_avg_ndcg
    
    def bayesian_dcg(self, predicted_scores, relevance_scores, k):
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevance_scores = relevance_scores[sorted_indices[:k]]
                
        dcg_value = np.sum(sorted_relevance_scores / np.log2(np.arange(2, len(sorted_relevance_scores) + 2)))
        return dcg_value

    def bayesian_ndcg(self, predicted_scores, relevance_scores, k):
        predicted_scores = np.ravel(predicted_scores)
        relevance_scores = np.ravel(relevance_scores)
        
        k = min(k, len(predicted_scores))
        
        mask = ~np.isnan(relevance_scores)
        predicted_scores = predicted_scores[mask]
        relevance_scores = relevance_scores[mask]
        
        ideal_relevance_scores = np.sort(relevance_scores)[::-1][:k]
        
        idcg = self.bayesian_dcg(ideal_relevance_scores, ideal_relevance_scores, k)
        
        dcg = self.bayesian_dcg(predicted_scores, relevance_scores, k)

        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_ndcg_for_all_users(self, predicted_scores, relevance_scores, k):
        ndcg_sum = 0.0
        num_users = predicted_scores.shape[0]

        for i in range(num_users):
            ndcg_sum += self.bayesian_ndcg(predicted_scores[i], relevance_scores[i], k)

        return ndcg_sum / num_users