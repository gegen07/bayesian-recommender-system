import pickle
from preprocess import preprocess_data, split_train_test

def preprocess(input_data_user: str, input_data_movie: str, output_train_data: str, output_test_data: str):
    dense_data = preprocess_data(input_data_user, input_data_movie)
    train, test = split_train_test(dense_data)
    
    with open(output_train_data, "wb") as f:
        pickle.dump(train, f)
    with open(output_test_data, "wb") as f:
        pickle.dump(test, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-user', type=str)
    parser.add_argument('--input-data-movie', type=str)
    parser.add_argument('--output-train-data', type=str)
    parser.add_argument('--output-test-data', type=str)

    args = parser.parse_args()

    preprocess(args.input_data_user, args.input_data_movie, args.output_train_data, args.output_test_data)