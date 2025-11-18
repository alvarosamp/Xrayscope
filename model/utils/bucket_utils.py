import time 
import botocore as bt

def wait_for_bucket_deletion(s3_client, bucket_name, timeout = 60):
    """
    Waits for an S3 bucket to be deleted.

    Args:
        s3_client: Boto3 S3 client.
        bucket_name: Name of the S3 bucket to wait for deletion.
        timeout: Maximum time to wait in seconds (default is 60 seconds).

    Raises:
        TimeoutError: If the bucket is not deleted within the specified timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            s3_client.head_bucket(Bucket = bucket_name)
            print(f'Bucket {bucket_name} is available')
            return True
        except bt.exceptions.ClientError as e:
            error_code = int(e.response['Error']['Code'])
            time.sleep(5)
    return False

def load_images_from_bucket(s3, bucket, prefix, label,img_size):
    """
    Load images from an S3 bucket.

    Args:
        s3: Boto3 S3 client.
        bucket: Name of the S3 bucket.
        prefix: Prefix (folder path) in the S3 bucket.
        label: Label to assign to the images.
        img_size: Size to which images should be resized.
    """
    import cv2
    import numpy as np

    data = []
    response = s3.list_objects_v2(Bucket = bucket, Prefix = prefix)
    for obj in response.get('Contents', []):
        key = obj["Key"]
        if key.endswith('/'):
            continue  # Pula pastas
        obj_response = s3.get_object(Bucket = bucket, Key = key)
        file_bytes = obj_response['Body'].read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, img_size)
            img_flat = img.flatten()
            data.append((img_flat, label))
        return data 