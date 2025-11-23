import time
import botocore as bt


def wait_for_bucket_deletion(s3_client, bucket_name, timeout=60):
    """Waits for an S3 bucket to be deleted.

    Args:
        s3_client: Boto3 S3 client.
        bucket_name: Name of the S3 bucket to wait for deletion.
        timeout: Maximum time to wait in seconds (default: 60).

    Returns:
        True if the bucket was found (head_bucket succeeded) within the timeout,
        False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            # If head_bucket succeeds, the bucket exists / is available
            print(f'Bucket {bucket_name} is available')
            return True
        except bt.exceptions.ClientError:
            # Bucket not available yet; wait and retry
            time.sleep(5)
    return False


def wait_for_bucket(s3_client, bucket_name, timeout=60):
    """Wait until an S3 bucket exists and is accessible.

    This function matches the name expected by callers in the codebase.

    Args:
        s3_client: Boto3 S3 client.
        bucket_name: Name of the bucket to wait for.
        timeout: Maximum wait time in seconds.

    Returns:
        True if the bucket became available within timeout, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            return True
        except bt.exceptions.ClientError:
            time.sleep(2)
    return False


def load_images_from_bucket(s3, bucket, prefix, label, img_size):
    """Load images from an S3 bucket.

    Args:
        s3: Boto3 S3 client.
        bucket: Name of the S3 bucket.
        prefix: Prefix (folder path) in the S3 bucket.
        label: Label to assign to the images.
        img_size: Size to which images should be resized (width, height).

    Returns:
        A list of tuples (flattened_image_array, label).
    """
    import cv2
    import numpy as np

    data = []
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response.get('Contents', []):
        key = obj.get("Key")
        if not key or key.endswith('/'):
            continue  # skip folders or invalid keys
        obj_response = s3.get_object(Bucket=bucket, Key=key)
        file_bytes = obj_response['Body'].read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, img_size)
            img_flat = img.flatten()
            data.append((img_flat, label))

    return data