import pandas as pd
import numpy as np
import pymc as pm
import argparse

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

def preprocess_data(input_data_user, input_data_movie):
    data_kwargs = dict(sep="\t", names=["userid", "itemid", "rating", "timestamp"])

    try:
        data = pd.read_csv(input_data_user, **data_kwargs)
    except FileNotFoundError:
        data = pd.read_csv(pm.get_data(input_data_user), **data_kwargs)
        


    movie_columns  = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 
                    'unknown','Action','Adventure', 'Animation',"Children's", 'Comedy', 'Crime',
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    # fmt: on

    item_kwargs = dict(sep="|", names=movie_columns, index_col="movie id", parse_dates=["release date"])
    try:
        movies = pd.read_csv(input_data_movie, **item_kwargs, encoding='latin-1')
    except FileNotFoundError:
        movies = pd.read_csv(pm.get_data(input_data_movie), **item_kwargs)
        
    dense_data = data.pivot(index="userid", columns="itemid", values="rating").values
    
    return dense_data
    
def split_train_test(data, percent_test=0.1):
    """Split the data into train/test sets.
    :param int percent_test: Percentage of data to use for testing. Default 10.
    """
    n, m = data.shape  # # users, # movies
    N = n * m  # # cells in matrix

    # Prepare train/test ndarrays.
    train = data.copy()
    test = np.ones(data.shape) * np.nan

    # Draw random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))  # ignore nan values in data
    idx_pairs = list(zip(tosample[0], tosample[1]))  # tuples of row/col index pairs

    test_size = int(len(idx_pairs) * percent_test)  # use 10% of data as test set
    train_size = len(idx_pairs) - test_size  # and remainder for training

    indices = np.arange(len(idx_pairs))  # indices of index pairs
    sample = rng.choice(indices, replace=False, size=test_size)

    # Transfer random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan  # remove from train set

    # Verify everything worked properly
    assert train_size == N - np.isnan(train).sum()
    assert test_size == N - np.isnan(test).sum()

    # Return train set and test set
    return train, test
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_user', type=str, help="Path to the input data")
    parser.add_argument('--input_data_movie', type=str, help="Path to the input data")
    parser.add_argument('--output_train_data', type=str, help="Path to the output data")
    parser.add_argument('--output_test_data', type=str, help="Path to the output data")
    
    args = parser.parse_args()
    

    dense_data = preprocess_data(args.input_data_user, args.input_data_movie)
    train, test = split_train_test(dense_data)
    
    import pickle
    
    with open(args.output_train_data, "wb") as f:
        pickle.dump(train, f)

    with open(args.output_test_data, "wb") as f:
        pickle.dump(test, f)
