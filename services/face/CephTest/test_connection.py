#!/usr/bin/env python3
import boto3
import requests
import sys

# Disable SSL warnings
requests.packages.urllib3.disable_warnings()

def test_connection(endpoint_url):
    """Test basic connectivity to endpoint."""
    try:
        print(f"Testing connection to: {endpoint_url}")
        response = requests.get(endpoint_url, verify=False, timeout=10)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("✓ Endpoint is reachable")
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Connection failed: {e}")

def test_s3_connection(endpoint_url, access_key, secret_key):
    """Test S3 connection."""
    try:
        print("\nTesting S3 connection...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name='us-east-1',
            use_ssl=endpoint_url.startswith('https'),
            verify=False
        )

        print("Attempting to list buckets...")
        response = s3_client.list_buckets()
        buckets = response.get('Buckets', [])
        print(f"✓ Success! Found {len(buckets)} bucket(s)")
        for bucket in buckets:
            print(f"  - {bucket['Name']} (created: {bucket['CreationDate']})")

    except Exception as e:
        print(f"✗ S3 connection failed: {e}")

if __name__ == "__main__":
    # Test different endpoint configurations
    endpoints_to_test = [
        "https://10.115.74.180:8443",
        "http://10.115.74.180:8080",
        "https://10.115.74.180:8443/rgw",
        "http://10.115.74.180:8080/rgw"
    ]

    access_key = "L95EEYTE5BXJDZ0ASCM0"
    secret_key = "rg9LmlDwKlflAqCiacV0eY1y5f2jJfFuviHb67pX"

    print("=== Ceph Endpoint Testing ===\n")

    for endpoint in endpoints_to_test:
        print(f"\n--- Testing: {endpoint} ---")
        test_connection(endpoint)
        test_s3_connection(endpoint, access_key, secret_key)
