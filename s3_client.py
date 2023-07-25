"""
This module implements a S3Client class which acts as a connector
to retrieve files from S3 buckets using boto3 library
"""

import boto3
from retry import retry
from botocore.exceptions import ClientError

import os
import logging
from prefect import task

logger = logging.getLogger(__name__)


@task(name="S3 Download data", log_prints=True)
@retry(tries=3, delay=30)
def download_data(
    aws_key: str, aws_secret: str, bucket: str, prefix: str, output_name: str
) -> None:
    client = boto3.client(
        "s3", aws_access_key_id=aws_key, aws_secret_access_key=aws_secret
    )

    try:
        client.download_file(bucket, prefix, output_name)
        logger.info("File downloaded from '%s' to '%s'", bucket, output_name)
    except Exception as error:
        logger.error("Error while trying to download dataset: %s", str(error))


@task(name="S3 Upload Model pickle", log_prints=True)
@retry(tries=3, delay=30)
def upload_data(
    aws_key: str, aws_secret: str, file_name: str, bucket: str, object_name: str = None
):
    client = boto3.client(
        "s3", aws_access_key_id=aws_key, aws_secret_access_key=aws_secret
    )
    try:
        client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logger.error(e)
        return False
