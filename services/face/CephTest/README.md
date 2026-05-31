# Ceph Object Storage CLI

A Python-based command-line interface for interacting with Ceph object storage using the S3-compatible API.

## Features

- **Bucket Management**: Create, list, and delete buckets
- **File Operations**: Upload, download, list, and delete files
- **Metadata Support**: Attach and retrieve custom metadata
- **Configuration Management**: YAML-based configuration with validation
- **Comprehensive Logging**: File and console logging with configurable levels
- **Error Handling**: Robust error handling with clear error messages
- **CLI Interface**: Easy-to-use command-line interface with argparse

## Project Structure

```
CephTest/
├── config.yaml          # Configuration file (create from template)
├── main.py             # CLI entry point
├── ceph_client.py      # Ceph operations implementation
├── utils.py            # Configuration and logging utilities
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Access to a Ceph object storage cluster
- Ceph access credentials (access key and secret key)

### Setup Steps

1. **Clone or download the project**:
   ```bash
   cd d:/ATG/scripts/CephTest
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your Ceph credentials**:
   
   Edit `config.yaml` and replace the placeholder values with your actual Ceph credentials:
   
   ```yaml
   auth:
     access_key: "YOUR_ACCESS_KEY_HERE"
     secret_key: "YOUR_SECRET_KEY_HERE"
   
   endpoint:
     url: "https://your-ceph-endpoint.com"
     region: "us-east-1"
     use_ssl: true
     verify_ssl: true  # Set to false for self-signed certificates
   
   bucket:
     default_name: "my-default-bucket"
   ```

## Usage

### Basic Command Structure

```bash
python main.py <command> [options]
```

### Available Commands

#### 1. List All Buckets

```bash
python main.py list-buckets
```

**Output**:
```
Found 3 bucket(s):

+---------------+---------------------+
| Bucket Name   | Creation Date       |
+===============+=====================+
| my-bucket-1   | 2024-01-15 10:30:45 |
| my-bucket-2   | 2024-01-16 14:22:10 |
| test-bucket   | 2024-01-20 09:15:33 |
+---------------+---------------------+
```

#### 2. Create a Bucket

```bash
python main.py create-bucket --bucket my-new-bucket
```

**Output**:
```
✓ Bucket 'my-new-bucket' created successfully
```

#### 3. Delete a Bucket

```bash
# With confirmation prompt
python main.py delete-bucket --bucket my-old-bucket

# Force delete (removes all objects first)
python main.py delete-bucket --bucket my-old-bucket --force
```

#### 4. List Files in a Bucket

```bash
# List all files in default bucket
python main.py list-files

# List files in specific bucket
python main.py list-files --bucket my-bucket

# List files with prefix filter
python main.py list-files --bucket my-bucket --prefix images/
```

**Output**:
```
Found 5 file(s) in bucket 'my-bucket':

+------------------+-----------+---------------------+
| File Name        | Size      | Last Modified       |
+==================+===========+=====================+
| document.pdf     | 2.45 MB   | 2024-01-22 11:30:00 |
| image.jpg        | 856.32 KB | 2024-01-22 12:15:22 |
| data.csv         | 125.67 KB | 2024-01-22 13:45:10 |
+------------------+-----------+---------------------+
```

#### 5. Upload a File

```bash
# Upload to default bucket with original filename
python main.py upload --file /path/to/file.txt

# Upload to specific bucket
python main.py upload --file /path/to/file.txt --bucket my-bucket

# Upload with custom name
python main.py upload --file /path/to/file.txt --bucket my-bucket --name custom-name.txt

# Upload with metadata
python main.py upload --file /path/to/file.txt --bucket my-bucket --metadata author=john version=1.0
```

**Output**:
```
✓ File uploaded successfully to 'my-bucket/file.txt'
```

#### 6. Download a File

```bash
# Download to current directory with original name
python main.py download --object file.txt

# Download from specific bucket
python main.py download --object file.txt --bucket my-bucket

# Download with custom output path
python main.py download --object file.txt --bucket my-bucket --output /path/to/save/newname.txt
```

**Output**:
```
✓ File downloaded successfully to '/path/to/save/newname.txt'
```

#### 7. Delete a File

```bash
# With confirmation prompt
python main.py delete-file --object file.txt --bucket my-bucket

# Skip confirmation
python main.py delete-file --object file.txt --bucket my-bucket --force
```

**Output**:
```
✓ File 'file.txt' deleted successfully from bucket 'my-bucket'
```

#### 8. Get Object Information

```bash
python main.py info --object file.txt --bucket my-bucket
```

**Output**:
```
Object Information for 'my-bucket/file.txt':

+------------------+---------------------+
| Property         | Value               |
+==================+=====================+
| Content Length   | 1024                |
| Content Type     | text/plain          |
| Last Modified    | 2024-01-22 11:30:00 |
| ETag             | abc123def456        |
+------------------+---------------------+

Custom Metadata:
+--------+-------+
| Key    | Value |
+========+=======+
| author | john  |
| version| 1.0   |
+--------+-------+
```

### Using Custom Configuration File

```bash
python main.py --config /path/to/custom-config.yaml list-buckets
```

## Configuration Reference

### Authentication (`auth`)

- `access_key`: Your Ceph access key (required)
- `secret_key`: Your Ceph secret key (required)

### Endpoint (`endpoint`)

- `url`: Ceph endpoint URL (required)
- `region`: AWS region name (optional, default: "us-east-1")
- `use_ssl`: Use SSL/TLS connection (optional, default: true)
- `verify_ssl`: Verify SSL certificates (optional, default: true)

### Bucket (`bucket`)

- `default_name`: Default bucket name for operations (optional)
- `create_if_not_exists`: Auto-create default bucket (optional, default: false)

### Logging (`logging`)

- `level`: Log level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
- `log_file`: Path to log file (default: "ceph_operations.log")
- `console_output`: Enable console logging (default: true)

### Upload (`upload`)

- `chunk_size`: Chunk size for multipart uploads in bytes (default: 8388608)
- `max_concurrency`: Maximum concurrent upload threads (default: 10)

### Retry (`retry`)

- `max_attempts`: Maximum retry attempts for failed operations (default: 3)
- `backoff_factor`: Exponential backoff factor (default: 2)

## Logging

All operations are logged to both console and file (by default). Logs include:

- Operation timestamps
- Success/failure status
- Detailed error messages
- File sizes and transfer information

Log file location: `ceph_operations.log` (configurable in `config.yaml`)

## Error Handling

The CLI provides clear error messages for common issues:

- **Invalid credentials**: Authentication failed
- **Connection errors**: Cannot reach Ceph endpoint
- **Missing files**: File not found errors
- **Bucket errors**: Bucket doesn't exist or already exists
- **Configuration errors**: Missing or invalid configuration

## Troubleshooting

### SSL Certificate Errors

If you're using self-signed certificates, set `verify_ssl: false` in `config.yaml`:

```yaml
endpoint:
  verify_ssl: false
```

### Connection Timeout

Check your endpoint URL and ensure the Ceph cluster is accessible:

```bash
# Test connectivity
curl https://your-ceph-endpoint.com
```

### Authentication Errors

Verify your access key and secret key are correct in `config.yaml`.

### Permission Errors

Ensure your Ceph user has the necessary permissions for the operations you're attempting.

## Examples

### Complete Workflow Example

```bash
# 1. Create a new bucket
python main.py create-bucket --bucket my-project

# 2. Upload files
python main.py upload --file document.pdf --bucket my-project
python main.py upload --file image.jpg --bucket my-project --metadata type=image

# 3. List uploaded files
python main.py list-files --bucket my-project

# 4. Get file information
python main.py info --object document.pdf --bucket my-project

# 5. Download a file
python main.py download --object document.pdf --bucket my-project --output ./downloads/doc.pdf

# 6. Delete a file
python main.py delete-file --object image.jpg --bucket my-project --force

# 7. Delete the bucket
python main.py delete-bucket --bucket my-project --force
```

## Development

### Adding New Features

1. Add new methods to `CephClient` class in `ceph_client.py`
2. Create handler function in `main.py`
3. Add command parser in `main()` function
4. Update this README with usage examples

### Running Tests

```bash
# Test configuration loading
python -c "from utils import ConfigLoader; c = ConfigLoader(); print(c.load())"

# Test connection
python main.py list-buckets
```

## Dependencies

- **boto3**: AWS SDK for Python (S3-compatible operations)
- **botocore**: Low-level interface to AWS services
- **PyYAML**: YAML parser for configuration files
- **tabulate**: Pretty-print tabular data

## License

This project is provided as-is for interacting with Ceph object storage systems.

## Support

For issues related to:
- **Ceph cluster**: Contact your Ceph administrator
- **This CLI tool**: Check logs in `ceph_operations.log` for detailed error information

## Notes

- Default bucket name can be configured in `config.yaml` to avoid specifying `--bucket` for every command
- All file sizes are displayed in human-readable format (B, KB, MB, GB, TB)
- Operations are logged with timestamps for audit purposes
- The tool uses boto3's S3-compatible API, making it compatible with any S3-compatible storage system
