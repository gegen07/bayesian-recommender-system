from google.cloud import aiplatform
from kfp import compiler, Client
from kfp.dsl import pipeline

# Define the pipeline path (yaml file)
pipeline_package_path = 'gs://pmf-training/pmf_training_pipeline_with_preprocessing.yaml'

# Set up the project and region
project_id = "bayesian-neural-network-443600"
region = "us-central1"  # Change to your preferred region

# Initialize the Vertex AI platform
aiplatform.init(project=project_id, location=region)

# Define the pipeline job submission
def submit_pipeline():
    # Define the pipeline job name and arguments

    job_name = "pmf-training-pipeline"
    bucket_name = 'pmf-training'
    input_data_user = f'gs://{bucket_name}/u.data'
    input_data_movie = f'gs://{bucket_name}/u.item'

    # Define pipeline arguments
    pipeline_args = {
        "input_data_user": input_data_user,
        "input_data_movie": input_data_movie,
        "bucket_name": bucket_name,
    }

    # Submit the pipeline
    pipeline_job = aiplatform.PipelineJob(
        display_name=job_name,
        template_path=pipeline_package_path,
        pipeline_root="gs://bayesian-neural-network-443600/",
        parameter_values=pipeline_args,
    )

    # Run the pipeline job
    pipeline_job.run(sync=True)

if __name__ == "__main__":
    submit_pipeline()
