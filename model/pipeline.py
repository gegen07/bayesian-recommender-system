import kfp
from kfp.dsl import component, Output, Input, Dataset, pipeline, Metrics

# Define preprocessing component using command
@component(
    base_image="gcr.io/bayesian-neural-network-443600/preprocessing-pmf:latest",
)
def preprocessing_task(input_data_user: str, input_data_movie: str, 
                  output_train_data: Output[Dataset], output_test_data: Output[Dataset]):
    import shutil
    import subprocess

    command = [
        "python3", "main.py",
        "--input_data_user", input_data_user,
        "--input_data_movie", input_data_movie,
        "--output_train_data", "/tmp/train_data.pkl",
        "--output_test_data", "/tmp/test_data.pkl"
    ]
    
    subprocess.run(command, check=True)

    shutil.move("/tmp/train_data.pkl", output_train_data.path)
    shutil.move("/tmp/test_data.pkl", output_test_data.path)

@component(
    base_image="gcr.io/bayesian-neural-network-443600/pmf:latest",
)
def training_task(train_data: Input[Dataset], model_output: Output[Dataset]):
    import shutil
    import subprocess

    command = [
        "python3", "train_pmf.py",
        "--input-file", train_data.path,
        "--output-file", "/tmp/model_output.pkl"
    ]

    subprocess.run(command, check=True)

    shutil.move("/tmp/model_output.pkl", model_output.path)


@component(
    base_image="gcr.io/bayesian-neural-network-443600/pmf:latest",
)
def evaluation_task(train_data: Input[Dataset], test_data: Input[Dataset], model_input: Input[Dataset], metrics_output: Output[Metrics]):
    import json
    import subprocess

    command = [
        "python3", "evaluation.py",
        "--train-data", train_data.path,
        "--test-data", test_data.path,
        "--model-input", model_input.path,
        "--output-file", '/tmp/metrics.json'
    ]

    subprocess.run(command, check=True)

    output_metrics_dict = {}
    with open('/tmp/metrics.json', 'r') as f:
        output_metrics_dict = json.load(f)

    metrics_output.log_metric("train_rmse", output_metrics_dict["train_rmse"])
    metrics_output.log_metric("test_rmse", output_metrics_dict["test_rmse"])
    metrics_output.log_metric("train_avg_ndcg@5", output_metrics_dict["train_avg_ndcg@5"])
    metrics_output.log_metric("test_avg_ndcg@5", output_metrics_dict["test_avg_ndcg@5"])
    metrics_output.log_metric("train_avg_ndcg@3", output_metrics_dict["train_avg_ndcg@3"])
    metrics_output.log_metric("test_avg_ndcg@3", output_metrics_dict["test_avg_ndcg@3"])
    metrics_output.log_metric("train_avg_ndcg@2", output_metrics_dict["train_avg_ndcg@2"])
    metrics_output.log_metric("test_avg_ndcg@2", output_metrics_dict["test_avg_ndcg@2"])

@component(
    base_image="gcr.io/bayesian-neural-network-443600/upload-model:latest",
)
def upload_to_gcs(model_file: Input[Dataset], bucket_name: str, destination_blob_name: str):
    import subprocess

    command = [
        "python", "main.py",
        "--local_file", model_file.path,
        "--bucket_name", bucket_name,
        "--destination_blob_name", destination_blob_name
    ]

    subprocess.run(command, check=True)

@pipeline(
    name="PMF-Training-Pipeline-With-Preprocessing",
    description="A pipeline with preprocessing and PMF model training"
)
def pmf_pipeline_with_preprocessing(input_data_user: str, input_data_movie: str, bucket_name: str):
    
    preprocessing_op = preprocessing_task(input_data_user=input_data_user,
                                     input_data_movie=input_data_movie)
    preprocessing_op.set_caching_options(False)
    
    training_op = training_task(train_data=preprocessing_op.outputs['output_train_data'])
    training_op.set_caching_options(False)

    upload_op = upload_to_gcs(model_file=training_op.outputs['model_output'],
                              bucket_name=bucket_name,
                              destination_blob_name="pmf-model.pkl")
    upload_op.set_caching_options(False)
    
    evaluation_op = evaluation_task(train_data=preprocessing_op.outputs['output_train_data'],
                                    test_data=preprocessing_op.outputs['output_test_data'],
                                    model_input=training_op.outputs['model_output'])
    evaluation_op.set_caching_options(False)
    
    upload_op.after(training_op)
    evaluation_op.after(upload_op)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pmf_pipeline_with_preprocessing, "pmf_training_pipeline_with_preprocessing.yaml")
