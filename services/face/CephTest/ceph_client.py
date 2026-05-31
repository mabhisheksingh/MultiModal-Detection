import boto3
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
import os
from typing import List, Dict, Optional, Any
import logging
import uuid
import mimetypes
from .utils import format_size


class CephClient:
    """Client for interacting with Ceph object storage using S3-compatible API."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Ceph client.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.s3_client = None
        self.s3_resource = None
        self._initialize_client()
    
    def _detect_content_type(self, file_path: str) -> str:
        """
        Detect the MIME type of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string (defaults to 'application/octet-stream' if unknown)
        """
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            # Check if file is text or binary by reading first few bytes
            try:
                with open(file_path, 'rb') as f:
                    data = f.read(1024)
                    # Check if data is likely text (no null bytes)
                    if b'\x00' not in data:
                        content_type = 'text/plain'
                    else:
                        content_type = 'application/octet-stream'
            except Exception:
                content_type = 'application/octet-stream'
        
        return content_type
    
    def _initialize_client(self):
        """Initialize boto3 S3 client and resource with Ceph configuration."""
        try:
            auth = self.config['auth']
            endpoint = self.config['endpoint']
            
            boto_config = Config(
                signature_version='s3v4',
                retries={
                    'max_attempts': self.config.get('retry', {}).get('max_attempts', 3),
                    'mode': 'standard'
                }
            )
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=auth['access_key'],
                aws_secret_access_key=auth['secret_key'],
                endpoint_url=endpoint['url'],
                region_name=endpoint.get('region', 'us-east-1'),
                use_ssl=endpoint.get('use_ssl', True),
                verify=endpoint.get('verify_ssl', True),
                config=boto_config
            )
            
            self.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=auth['access_key'],
                aws_secret_access_key=auth['secret_key'],
                endpoint_url=endpoint['url'],
                region_name=endpoint.get('region', 'us-east-1'),
                use_ssl=endpoint.get('use_ssl', True),
                verify=endpoint.get('verify_ssl', True),
                config=boto_config
            )
            
            self.logger.info("Successfully initialized Ceph S3 client")
            
        except KeyError as e:
            self.logger.error(f"Missing configuration key: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Ceph client: {e}")
            raise
    
    def list_buckets(self) -> List[Dict[str, Any]]:
        """
        List all buckets in the Ceph storage.
        
        Returns:
            List of dictionaries containing bucket information
            
        Raises:
            Exception: If listing buckets fails
        """
        try:
            self.logger.info("Listing all buckets")
            response = self.s3_client.list_buckets()
            
            buckets = []
            for bucket in response.get('Buckets', []):
                buckets.append({
                    'name': bucket['Name'],
                    'creation_date': bucket['CreationDate']
                })
            
            self.logger.info(f"Found {len(buckets)} bucket(s)")
            return buckets
            
        except NoCredentialsError:
            self.logger.error("Invalid credentials")
            raise Exception("Authentication failed: Invalid credentials")
        except EndpointConnectionError:
            self.logger.error("Cannot connect to endpoint")
            raise Exception("Connection failed: Cannot reach Ceph endpoint")
        except ClientError as e:
            self.logger.error(f"Error listing buckets: {e}")
            raise Exception(f"Failed to list buckets: {e}")
    
    def create_bucket(self, bucket_name: str) -> bool:
        """
        Create a new bucket.
        
        Args:
            bucket_name: Name of the bucket to create
            
        Returns:
            True if successful
            
        Raises:
            Exception: If bucket creation fails
        """
        try:
            self.logger.info(f"Creating bucket: {bucket_name}")
            self.s3_client.create_bucket(Bucket=bucket_name)
            self.logger.info(f"Successfully created bucket: {bucket_name}")
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'BucketAlreadyOwnedByYou':
                self.logger.warning(f"Bucket already exists: {bucket_name}")
                return True
            elif error_code == 'BucketAlreadyExists':
                self.logger.error(f"Bucket name already taken: {bucket_name}")
                raise Exception(f"Bucket name '{bucket_name}' is already taken")
            else:
                self.logger.error(f"Error creating bucket: {e}")
                raise Exception(f"Failed to create bucket: {e}")
    
    def delete_bucket(self, bucket_name: str, force: bool = False) -> bool:
        """
        Delete a bucket.
        
        Args:
            bucket_name: Name of the bucket to delete
            force: If True, delete all objects in bucket first
            
        Returns:
            True if successful
            
        Raises:
            Exception: If bucket deletion fails
        """
        try:
            self.logger.info(f"Deleting bucket: {bucket_name}")
            
            if force:
                self.logger.info(f"Force delete enabled, removing all objects first")
                bucket = self.s3_resource.Bucket(bucket_name)
                bucket.objects.all().delete()
            
            self.s3_client.delete_bucket(Bucket=bucket_name)
            self.logger.info(f"Successfully deleted bucket: {bucket_name}")
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                self.logger.error(f"Bucket does not exist: {bucket_name}")
                raise Exception(f"Bucket '{bucket_name}' does not exist")
            elif error_code == 'BucketNotEmpty':
                self.logger.error(f"Bucket is not empty: {bucket_name}")
                raise Exception(
                    f"Bucket '{bucket_name}' is not empty. Use --force to delete all objects first"
                )
            else:
                self.logger.error(f"Error deleting bucket: {e}")
                raise Exception(f"Failed to delete bucket: {e}")
    
    def list_files(self, bucket_name: str, prefix: str = "") -> List[Dict[str, Any]]:
        """
        List files in a bucket.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter files
            
        Returns:
            List of dictionaries containing file information
            
        Raises:
            Exception: If listing files fails
        """
        try:
            self.logger.info(f"Listing files in bucket: {bucket_name}")
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            files = []
            for page in pages:
                for obj in page.get('Contents', []):
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'size_formatted': format_size(obj['Size']),
                        'last_modified': obj['LastModified'],
                        'etag': obj['ETag'].strip('"')
                    })
            
            self.logger.info(f"Found {len(files)} file(s) in bucket: {bucket_name}")
            return files
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                self.logger.error(f"Bucket does not exist: {bucket_name}")
                raise Exception(f"Bucket '{bucket_name}' does not exist")
            else:
                self.logger.error(f"Error listing files: {e}")
                raise Exception(f"Failed to list files: {e}")
    
    def upload_file(
        self, 
        file_path: str, 
        bucket_name: str, 
        object_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload a file to a bucket.
        
        Args:
            file_path: Path to the file to upload
            bucket_name: Name of the bucket
            object_name: S3 object name (defaults to file basename)
            metadata: Optional metadata to attach to the object
            
        Returns:
            True if successful
            
        Raises:
            Exception: If upload fails
        """
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        try:
            file_size = os.path.getsize(file_path)
            content_type = self._detect_content_type(file_path)
            
            self.logger.info(
                f"Uploading file: {file_path} ({format_size(file_size)}) "
                f"to {bucket_name}/{object_name} (Content-Type: {content_type})"
            )
            
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Set ContentType
            extra_args['ContentType'] = content_type
            
            self.s3_client.upload_file(
                file_path,
                bucket_name,
                object_name,
                ExtraArgs=extra_args if extra_args else None
            )
            
            self.logger.info(f"Successfully uploaded: {object_name}")
            return True
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise Exception(f"File not found: {file_path}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                self.logger.error(f"Bucket does not exist: {bucket_name}")
                raise Exception(f"Bucket '{bucket_name}' does not exist")
            else:
                self.logger.error(f"Error uploading file: {e}")
                raise Exception(f"Failed to upload file: {e}")
    
    def save_file_into_ceph(self, file_path: str, bucket_name: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """
        Upload a file to Ceph using UUID as object key, following Java saveFileIntoCeph pattern.
        
        Args:
            file_path: Path to the file to upload
            bucket_name: Name of the bucket
            metadata: Optional metadata to attach to the object
            
        Returns:
            Object store ID (UUID) used as the key
            
        Raises:
            Exception: If upload fails
        """
        object_store_id = str(uuid.uuid4())
        
        try:
            file_size = os.path.getsize(file_path)
            content_type = self._detect_content_type(file_path)
            
            self.logger.info(
                f"Uploading file: {file_path} ({format_size(file_size)}) "
                f"to {bucket_name}/{object_store_id} (Content-Type: {content_type})"
            )
            
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Set ContentType
            extra_args['ContentType'] = content_type
            
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=object_store_id,
                Body=open(file_path, 'rb'),
                **extra_args
            )
            
            self.logger.info(f"Successfully uploaded: {object_store_id}")
            return object_store_id
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise Exception(f"File not found: {file_path}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                self.logger.error(f"Bucket does not exist: {bucket_name}")
                raise Exception(f"Bucket '{bucket_name}' does not exist")
            else:
                self.logger.error(f"Error uploading file: {e}")
                raise Exception(f"Failed to upload file: {e}")
    
    def download_file(
        self, 
        bucket_name: str, 
        object_name: str, 
        file_path: Optional[str] = None
    ) -> str:
        """
        Download a file from a bucket.
        
        Args:
            bucket_name: Name of the bucket
            object_name: S3 object name to download
            file_path: Local path to save file (defaults to object name)
            
        Returns:
            Path to downloaded file
            
        Raises:
            Exception: If download fails
        """
        if file_path is None:
            file_path = object_name
        
        try:
            self.logger.info(
                f"Downloading {bucket_name}/{object_name} to {file_path}"
            )
            
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            
            self.s3_client.download_file(bucket_name, object_name, file_path)
            
            file_size = os.path.getsize(file_path)
            self.logger.info(
                f"Successfully downloaded: {file_path} ({format_size(file_size)})"
            )
            return file_path
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                self.logger.error(f"Bucket does not exist: {bucket_name}")
                raise Exception(f"Bucket '{bucket_name}' does not exist")
            elif error_code == '404':
                self.logger.error(f"Object not found: {object_name}")
                raise Exception(f"Object '{object_name}' not found in bucket")
            else:
                self.logger.error(f"Error downloading file: {e}")
                raise Exception(f"Failed to download file: {e}")
    
    def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """
        Delete a file from a bucket.
        
        Args:
            bucket_name: Name of the bucket
            object_name: S3 object name to delete
            
        Returns:
            True if successful
            
        Raises:
            Exception: If deletion fails
        """
        try:
            self.logger.info(f"Deleting {bucket_name}/{object_name}")
            
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_name)
            
            self.logger.info(f"Successfully deleted: {object_name}")
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                self.logger.error(f"Bucket does not exist: {bucket_name}")
                raise Exception(f"Bucket '{bucket_name}' does not exist")
            else:
                self.logger.error(f"Error deleting file: {e}")
                raise Exception(f"Failed to delete file: {e}")
    
    def get_object_metadata(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """
        Get metadata for an object.
        
        Args:
            bucket_name: Name of the bucket
            object_name: S3 object name
            
        Returns:
            Dictionary containing object metadata
            
        Raises:
            Exception: If operation fails
        """
        try:
            self.logger.info(f"Getting metadata for {bucket_name}/{object_name}")
            
            response = self.s3_client.head_object(Bucket=bucket_name, Key=object_name)
            
            metadata = {
                'content_length': response['ContentLength'],
                'content_type': response.get('ContentType', 'unknown'),
                'last_modified': response['LastModified'],
                'etag': response['ETag'].strip('"'),
                'metadata': response.get('Metadata', {})
            }
            
            return metadata
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                self.logger.error(f"Object not found: {object_name}")
                raise Exception(f"Object '{object_name}' not found in bucket")
            else:
                self.logger.error(f"Error getting metadata: {e}")
                raise Exception(f"Failed to get object metadata: {e}")
