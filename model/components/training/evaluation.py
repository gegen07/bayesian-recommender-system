from pmf import PMF
import json
import numpy as np
import pickle

def evaluation_task(train_data_file, test_data_file, model, metrics_output):

    # Load ground truth and predictions
    with open(train_data_file, "rb") as f:
        train_data = pickle.load(f)
    
    with open(test_data_file, "rb") as f:
        test_data = pickle.load(f)

    train_rmse, test_rmse, train_avg_ndcg, test_avg_ndcg = model.eval_map(train_data, test_data, k=5)

    _, _, train_avg_ndcg_3, test_avg_ndcg_3 = model.eval_map(train_data, test_data, k=3)
    _, _, train_avg_ndcg_2, test_avg_ndcg_2 = model.eval_map(train_data, test_data, k=2)

    metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_avg_ndcg@5": train_avg_ndcg,
        "test_avg_ndcg@5": test_avg_ndcg,
        "train_avg_ndcg@3": train_avg_ndcg_3,
        "test_avg_ndcg@3": test_avg_ndcg_3,
        "train_avg_ndcg@2": train_avg_ndcg_2,
        "test_avg_ndcg@2": test_avg_ndcg_2
    }

    print(metrics)

    with open(metrics_output, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    import argparse
    import cloudpickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-input', type=str)
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--output-file', type=str)

    args = parser.parse_args()

    model: PMF = None
    with open(args.model_input, "rb") as f:
        model = cloudpickle.load(f)

    evaluation_task(args.train_data, args.test_data, model, args.output_file)
