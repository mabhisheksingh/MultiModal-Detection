#!/usr/bin/env python3
import argparse
import sys
from typing import Optional
from utils import ConfigLoader, LoggerSetup, validate_file_path
from ceph_client import CephClient
from tabulate import tabulate


def print_success(message: str):
    """Print success message in green."""
    print(f"✓ {message}")


def print_error(message: str):
    """Print error message in red."""
    print(f"✗ Error: {message}", file=sys.stderr)


def print_table(data: list, headers: list):
    """Print data in a formatted table."""
    if not data:
        print("No data to display")
        return
    print(tabulate(data, headers=headers, tablefmt="grid"))


def handle_list_buckets(client: CephClient, args):
    """Handle list-buckets command."""
    try:
        buckets = client.list_buckets()
        
        if not buckets:
            print("No buckets found")
            return 0
        
        table_data = [
            [bucket['name'], bucket['creation_date'].strftime('%Y-%m-%d %H:%M:%S')]
            for bucket in buckets
        ]
        
        print(f"\nFound {len(buckets)} bucket(s):\n")
        print_table(table_data, ['Bucket Name', 'Creation Date'])
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def handle_create_bucket(client: CephClient, args):
    """Handle create-bucket command."""
    try:
        client.create_bucket(args.bucket)
        print_success(f"Bucket '{args.bucket}' created successfully")
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def handle_delete_bucket(client: CephClient, args):
    """Handle delete-bucket command."""
    try:
        if not args.force:
            confirm = input(f"Are you sure you want to delete bucket '{args.bucket}'? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Operation cancelled")
                return 0
        
        client.delete_bucket(args.bucket, force=args.force)
        print_success(f"Bucket '{args.bucket}' deleted successfully")
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def handle_list_files(client: CephClient, args):
    """Handle list-files command."""
    try:
        bucket = args.bucket or client.config.get('bucket', {}).get('default_name')
        if not bucket:
            print_error("No bucket specified and no default bucket configured")
            return 1
        
        files = client.list_files(bucket, prefix=args.prefix or "")
        
        if not files:
            print(f"No files found in bucket '{bucket}'")
            return 0
        
        table_data = [
            [
                file['key'],
                f"s3://{bucket}/{file['key']}",
                file['size_formatted'],
                file['last_modified'].strftime('%Y-%m-%d %H:%M:%S'),
                file['etag']
            ]
            for file in files
        ]
        
        print(f"\nFound {len(files)} file(s) in bucket '{bucket}':\n")
        print_table(table_data, ['File Name', 'Path', 'Size', 'Last Modified', 'Object ID'])
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def handle_upload(client: CephClient, args):
    """Handle upload command."""
    try:
        file_path = validate_file_path(args.file)
        
        bucket = args.bucket or client.config.get('bucket', {}).get('default_name')
        if not bucket:
            print_error("No bucket specified and no default bucket configured")
            return 1
        
        object_name = args.name if args.name else None
        
        metadata = {}
        if args.metadata:
            for item in args.metadata:
                if '=' in item:
                    key, value = item.split('=', 1)
                    metadata[key] = value
        
        client.upload_file(
            file_path,
            bucket,
            object_name=object_name,
            metadata=metadata if metadata else None
        )
        
        uploaded_name = object_name if object_name else file_path.split('/')[-1].split('\\')[-1]
        print_success(f"File uploaded successfully to '{bucket}/{uploaded_name}'")
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def handle_upload_uuid(client: CephClient, args):
    """Handle upload-uuid command using UUID as object key."""
    try:
        file_path = validate_file_path(args.file)
        
        bucket = args.bucket or client.config.get('bucket', {}).get('default_name')
        if not bucket:
            print_error("No bucket specified and no default bucket configured")
            return 1
        
        metadata = {}
        if args.metadata:
            for item in args.metadata:
                if '=' in item:
                    key, value = item.split('=', 1)
                    metadata[key] = value
        
        object_store_id = client.save_file_into_ceph(
            file_path,
            bucket,
            metadata=metadata if metadata else None
        )
        
        print_success(f"File uploaded successfully with Object ID: {object_store_id}")
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def handle_download(client: CephClient, args):
    """Handle download command."""
    try:
        bucket = args.bucket or client.config.get('bucket', {}).get('default_name')
        if not bucket:
            print_error("No bucket specified and no default bucket configured")
            return 1
        
        output_path = args.output if args.output else args.object
        
        downloaded_path = client.download_file(bucket, args.object, output_path)
        print_success(f"File downloaded successfully to '{downloaded_path}'")
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def handle_delete_file(client: CephClient, args):
    """Handle delete-file command."""
    try:
        bucket = args.bucket or client.config.get('bucket', {}).get('default_name')
        if not bucket:
            print_error("No bucket specified and no default bucket configured")
            return 1
        
        if not args.force:
            confirm = input(f"Are you sure you want to delete '{args.object}' from '{bucket}'? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Operation cancelled")
                return 0
        
        client.delete_file(bucket, args.object)
        print_success(f"File '{args.object}' deleted successfully from bucket '{bucket}'")
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def handle_info(client: CephClient, args):
    """Handle info command to get object metadata."""
    try:
        bucket = args.bucket or client.config.get('bucket', {}).get('default_name')
        if not bucket:
            print_error("No bucket specified and no default bucket configured")
            return 1
        
        metadata = client.get_object_metadata(bucket, args.object)
        
        print(f"\nObject Information for '{bucket}/{args.object}':\n")
        table_data = [
            ['Content Length', metadata['content_length']],
            ['Content Type', metadata['content_type']],
            ['Last Modified', metadata['last_modified'].strftime('%Y-%m-%d %H:%M:%S')],
            ['Object ID', metadata['etag']]
        ]
        
        print_table(table_data, ['Property', 'Value'])
        
        return 0
        
    except Exception as e:
        print_error(str(e))
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Ceph Object Storage CLI - Interact with Ceph using S3-compatible API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list-buckets
  %(prog)s create-bucket --bucket my-bucket
  %(prog)s list-files --bucket my-bucket
  %(prog)s upload --file /path/to/file.txt --bucket my-bucket
  %(prog)s upload --file /path/to/file.txt --bucket my-bucket --name custom-name.txt
  %(prog)s download --object file.txt --bucket my-bucket --output /path/to/save.txt
  %(prog)s delete-file --object file.txt --bucket my-bucket
  %(prog)s delete-bucket --bucket my-bucket --force
  %(prog)s info --object file.txt --bucket my-bucket
        """
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    subparsers.add_parser('list-buckets', help='List all buckets')
    
    create_bucket_parser = subparsers.add_parser('create-bucket', help='Create a new bucket')
    create_bucket_parser.add_argument('--bucket', required=True, help='Bucket name')
    
    delete_bucket_parser = subparsers.add_parser('delete-bucket', help='Delete a bucket')
    delete_bucket_parser.add_argument('--bucket', required=True, help='Bucket name')
    delete_bucket_parser.add_argument('--force', action='store_true', 
                                     help='Force delete (remove all objects first)')
    
    list_files_parser = subparsers.add_parser('list-files', help='List files in a bucket')
    list_files_parser.add_argument('--bucket', help='Bucket name (uses default if not specified)')
    list_files_parser.add_argument('--prefix', help='Filter files by prefix')
    
    upload_parser = subparsers.add_parser('upload', help='Upload a file to a bucket')
    upload_parser.add_argument('--file', required=True, help='Path to file to upload')
    upload_parser.add_argument('--bucket', help='Bucket name (uses default if not specified)')
    upload_parser.add_argument('--name', help='Object name in bucket (defaults to filename)')
    upload_parser.add_argument('--metadata', nargs='*', 
                              help='Metadata in key=value format (e.g., author=john version=1.0)')
    
    upload_uuid_parser = subparsers.add_parser('upload-uuid', help='Upload a file with UUID as object key')
    upload_uuid_parser.add_argument('--file', required=True, help='Path to file to upload')
    upload_uuid_parser.add_argument('--bucket', help='Bucket name (uses default if not specified)')
    upload_uuid_parser.add_argument('--metadata', nargs='*', 
                                   help='Metadata in key=value format (e.g., author=john version=1.0)')
    
    download_parser = subparsers.add_parser('download', help='Download a file from a bucket')
    download_parser.add_argument('--object', required=True, help='Object name to download')
    download_parser.add_argument('--bucket', help='Bucket name (uses default if not specified)')
    download_parser.add_argument('--output', help='Output file path (defaults to object name)')
    
    delete_file_parser = subparsers.add_parser('delete-file', help='Delete a file from a bucket')
    delete_file_parser.add_argument('--object', required=True, help='Object name to delete')
    delete_file_parser.add_argument('--bucket', help='Bucket name (uses default if not specified)')
    delete_file_parser.add_argument('--force', action='store_true', 
                                    help='Skip confirmation prompt')
    
    info_parser = subparsers.add_parser('info', help='Get object metadata')
    info_parser.add_argument('--object', required=True, help='Object name')
    info_parser.add_argument('--bucket', help='Bucket name (uses default if not specified)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()
        
        logger = LoggerSetup.setup_logger(config)
        
        client = CephClient(config, logger)
        
        commands = {
            'list-buckets': handle_list_buckets,
            'create-bucket': handle_create_bucket,
            'delete-bucket': handle_delete_bucket,
            'list-files': handle_list_files,
            'upload': handle_upload,
            'upload-uuid': handle_upload_uuid,
            'download': handle_download,
            'delete-file': handle_delete_file,
            'info': handle_info
        }
        
        handler = commands.get(args.command)
        if handler:
            return handler(client, args)
        else:
            print_error(f"Unknown command: {args.command}")
            return 1
            
    except FileNotFoundError as e:
        print_error(str(e))
        return 1
    except ValueError as e:
        print_error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
