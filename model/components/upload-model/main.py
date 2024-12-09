from google.cloud import storage

def upload_to_gcs(local_file, bucket_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_file', type=str)
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--destination_blob_name', type=str)

    args = parser.parse_args()

    upload_to_gcs(args.local_file, args.bucket_name, args.destination_blob_name)
