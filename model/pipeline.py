import kfp
from kfp.dsl import component, Output, Input, Dataset, pipeline

# Define preprocessing component using command
@component(
    base_image="gcr.io/bayesian-neural-network-443600/preprocessing-pmf:latest",
)
def preprocessing(input_data_user: str, input_data_movie: str, 
                  output_train_data: Output[Dataset], output_test_data: Output[Dataset]):
    import subprocess

    # Run the preprocessing script inside the container
    command = [
        "/opt/conda/bin/python", "main.py",
        "--input_data_user", input_data_user,
        "--input_data_movie", input_data_movie,
        "--output_train_data", "/tmp/train_data.pkl",
        "--output_test_data", "/tmp/test_data.pkl"
    ]
    
    subprocess.run(command, check=True)

    # Specify the output data file path
    output_train_data.path = '/tmp/train_data.pkl'
    output_test_data.path = '/tmp/test_data.pkl'

# Define PMF training component using command
@component(
    base_image="gcr.io/bayesian-neural-network-443600/pmf:latest",
)
def train_pmf(train_data: Input[Dataset], model_output: Output[Dataset]):
    import subprocess

    # Run the PMF model training script inside the container
    command = [
        "/opt/conda/bin/python", "train_pmf.py",
        "--train_data", train_data.path
    ]

    subprocess.run(command, check=True)

    # Specify the output model file path
    model_output.path = '/tmp/model_output.pkl'

# Define component to upload model to GCS using command
@component(
    base_image="gcr.io/bayesian-neural-network-443600/upload-model:latest",
)
def upload_to_gcs(model_file: Input[Dataset], bucket_name: str, destination_blob_name: str):
    import subprocess

    # Run the script to upload model to GCS
    command = [
        "python", "main.py",
        "--local_file", model_file.path,
        "--bucket_name", bucket_name,
        "--destination_blob_name", destination_blob_name
    ]

    subprocess.run(command, check=True)

# Define the pipeline function
@pipeline(
    name="PMF-Training-Pipeline-With-Preprocessing",
    description="A pipeline with preprocessing and PMF model training"
)
def pmf_pipeline_with_preprocessing(input_data_user: str, input_data_movie: str, bucket_name: str):
    
    # Ensure that output arguments are passed
    preprocessing_op = preprocessing(input_data_user=input_data_user,
                                     input_data_movie=input_data_movie)
    
    # Training operation depends on preprocessing output
    training_op = train_pmf(train_data=preprocessing_op.outputs['output_train_data'])

    # Upload operation depends on training output
    upload_op = upload_to_gcs(model_file=training_op.outputs['model_output'],
                              bucket_name=bucket_name,
                              destination_blob_name="pmf-model.pkl")
    
    # Define execution order
    upload_op.after(training_op)
    upload_op.after(preprocessing_op)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pmf_pipeline_with_preprocessing, "pmf_training_pipeline_with_preprocessing.yaml")
