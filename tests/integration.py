import sys
sys.path.append("..")

from model.components.preprocessing.preprocess import preprocess_data, split_train_test
from model.components.training.pmf import PMF
import pandas as pd
import numpy as np
import pymc as pm
import pytest

class TestPreprocessData:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.input_data_user = "data/u.data"
        self.input_data_movie = "data/u.item"
        self.train_data_file = "data/train.pkl"
        self.test_data_file = "data/test.pkl"

        import pickle

        with open(self.train_data_file, "rb") as f:
            self.train_data = pickle.load(f)
        
        with open(self.test_data_file, "rb") as f:
            self.test_data = pickle.load(f)


    def test_preprocess_data(self):
        dense_data = preprocess_data(self.input_data_user, self.input_data_movie)

        assert isinstance(dense_data, np.ndarray)
        

    def test_split_train_test(self):
        dense_data = preprocess_data(self.input_data_user, self.input_data_movie)
        train, test = split_train_test(dense_data)
        
        assert isinstance(train, np.ndarray)
        assert isinstance(test, np.ndarray)
        assert train.shape[1] == test.shape[1]
    
    def test_train(self):
        ALPHA = 2
        DIM = 10

        pmf = PMF(self.train_data_file, DIM, ALPHA, std=0.05)
        pmf.find_map()
        pmf.draw_samples(draws=5, tune=5)

        assert isinstance(pmf, PMF)
        assert hasattr(pmf, "_map")
    
    def test_train_output(self):
        ALPHA = 2
        DIM = 10

        pmf = PMF(self.train_data_file, DIM, ALPHA, std=0.05)
        pmf.find_map()
        pmf.draw_samples(draws=10, tune=10)
    
        train_rmse, test_rmse, train_avg_ndcg, test_avg_ndcg = pmf.eval_map(self.train_data, self.test_data, k=5)

        assert isinstance(train_rmse, float)
        assert isinstance(test_rmse, float)
        assert isinstance(train_avg_ndcg, float)
        assert isinstance(test_avg_ndcg, float)      