#!/usr/bin/env python3
"""
REST API OCR Text Extraction Benchmark

Tests OCR text extraction performance against labeled dataset.
Compares API predictions with ground truth labels.

Usage:
    python3 api_ocr_benchmark.py --api-url http://localhost:9003
    python3 api_ocr_benchmark.py --api-url http://localhost:9003 --dataset-path ../Indian\ vehicle\ license\ plate\ dataset/video_images
    python3 api_ocr_benchmark.py --api-url http://localhost:9003 --concurrency 3
"""

import os
import sys
import json
import time
import re
import argparse
import asyncio
from pathlib import Path
from collections import Counter
from statistics import mean, stdev
from datetime import datetime
import requests
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from typing import Dict, Any

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required for concurrent testing. Install with: pip install aiohttp")
    sys.exit(1)


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load key metrics from a saved benchmark JSON report."""
    data = json.loads(path.read_text())
    results = data.get("results", [])
    success = [r for r in results if r.get("status") == "success"]
    no_pred = [r for r in results if r.get("status") == "no_prediction"]
    errors = [r for r in results if r.get("status") == "error"]

    acc = [float(r.get("accuracy", 0.0)) for r in success]
    rt = [float(r.get("request_time", 0.0)) for r in success]

    m = {
        "mode": data.get("ocr_plate_text_mode", "unknown"),
        "file": str(path.name),
        "total": len(results),
        "success": len(success),
        "no_pred": len(no_pred),
        "errors": len(errors),
        "avg_accuracy_pct": round(mean(acc) * 100.0, 2) if acc else 0.0,
        "perfect": sum(1 for a in acc if a == 1.0),
        "gt80": sum(1 for a in acc if a > 0.8),
        "gt60": sum(1 for a in acc if a > 0.6),
        "gt50": sum(1 for a in acc if a > 0.5),
        "avg_time_s": round(mean(rt), 3) if rt else 0.0,
        "total_time_s": round(float(data.get("total_time", 0.0)), 2),
    }
    # Throughput = total images / total time
    m["throughput_img_per_s"] = round(m["total"] / m["total_time_s"], 2) if m["total_time_s"] else 0.0
    return m


def format_table(rows: list[Dict[str, Any]]) -> str:
    """Format a compact console table of selected metrics for multiple reports."""
    cols = [
        ("mode", "Mode"),
        ("file", "Report File"),
        ("total", "Total"),
        ("success", "Success"),
        ("no_pred", "No Pred"),
        ("errors", "Errors"),
        ("avg_accuracy_pct", "Avg Acc %"),
        ("perfect", "Perfect"),
        ("gt80", ">80%"),
        ("gt60", ">60%"),
        ("gt50", ">50%"),
        ("avg_time_s", "Avg Time (s)"),
        ("total_time_s", "Total Time (s)"),
        ("throughput_img_per_s", "Throughput (img/s)"),
    ]

    widths = []
    for key, header in cols:
        col_vals = [str(header)] + [str(r.get(key, "")) for r in rows]
        widths.append(max(len(v) for v in col_vals))

    lines = []
    header_cells = [str(h).ljust(w) for (k, h), w in zip(cols, widths)]
    lines.append(" | ".join(header_cells))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        cells = [str(r.get(k, "")).ljust(w) for (k, _), w in zip(cols, widths)]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def is_valid_indian_plate(plate_text):
    """Check if plate text exists (validation done by API via OCR_CLEANUP_REGEX in .env)."""
    if not plate_text:
        return False
    return True


def extract_label_from_filename(filename):
    """
    Extract license plate label from filename.
    Supports formats:
    - [PLATE_NUMBER]_[other_info].jpg (MP09CM0105_frame_0.jpg)
    - car-[type]-[PLATE_NUMBER]_[frame].jpg (car-wbs-MH01DE2780_00000.jpg)
    - video[N]_[frame].jpg (video10_1060.jpg)
    """
    basename = Path(filename).stem

    # Try format: car-type-PLATE_frame
    if basename.startswith('car-'):
        parts = basename.split('-')
        if len(parts) >= 3:
            potential_plate = parts[2].split('_')[0].upper()
            if is_valid_indian_plate(potential_plate):
                return potential_plate

    # Try format: PLATE_info
    parts = basename.split('_')
    if parts:
        potential_plate = parts[0].upper()
        if is_valid_indian_plate(potential_plate):
            return potential_plate

    return None


def extract_label_from_xml(image_path):
    """Extract license plate label from same-name XML annotation file."""
    xml_path = Path(image_path).with_suffix('.xml')
    if not xml_path.exists():
        return None

    try:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall('object'):
            name_node = obj.find('name')
            if name_node is not None and name_node.text:
                plate_text = name_node.text.strip().upper()
                if is_valid_indian_plate(plate_text):
                    return plate_text
    except Exception:
        return None

    return None


def find_image_files(dataset_path):
    """Find all image files in dataset."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return image_files

    for ext in image_extensions:
        image_files.extend(dataset_path.rglob(f'*{ext}'))

    return sorted(image_files)


import uuid


async def test_api_ocr_async(session, api_url, image_path, timeout=60, ocr_plate_text_mode='strict'):
    """
    Test OCR on API endpoint asynchronously using aiohttp.

    Args:
        session: aiohttp ClientSession
        api_url: Base URL of the API (e.g., http://localhost:9003)
        image_path: Path to image file
        timeout: Request timeout in seconds

    Returns:
        tuple: (API response dict, request_id, payload) or (None, request_id, payload) if error
    """
    try:
        # Resolve absolute path
        abs_image_path = os.path.abspath(image_path)
        request_id = str(uuid.uuid4())

        # Prepare the exact payload format requested
        payload = {
            "processing_config": {
                "confidence_threshold": 0.3,
                "frames_per_second": 10,
                "ocr_confidence_threshold": 0.5,
                "ocr_match_confidence": 0.85,
                "ocr_plate_text_mode": ocr_plate_text_mode,
                "global_id_match_score": 0.7,
                "nms_threshold": 0,
                "similarity_threshold": 0,
                "spatial_threshold": 0,
                "max_disappeared": 0,
                "confirmation_frames": 0,
                "save_cropped_faces": True,
                "generate_embeddings": True,
                "embedding_model": "string",
                "embedding_detector_backend": "string",
                "min_face_size": 0,
                "enable_face_alignment": True,
                "custom_output_fps": 0,
                "enable_cross_camera_reid": True,
                "cross_camera_threshold": 0,
                "platform": "string",
                "extra": {
                    "additionalProp1": {}
                }
            },
            "inputs": [
                {
                    "id": request_id,
                    "input_type": "image_file",
                    "options": {
                        "uri": abs_image_path,
                        "camera_id": "string",
                        "lat": -90,
                        "lon": -180,
                        "pixels_per_meter": 10
                    },
                    "metadata": {
                        "additionalProp1": {}
                    }
                }
            ]
        }

        url = urljoin(api_url, '/v2/anpr_object/process')
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": request_id
        }

        async with session.post(url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                return await response.json(), request_id, payload
            else:
                text = await response.text()
                print(f"API Error ({response.status}): {text}")
                return None, request_id, payload

    except asyncio.TimeoutError:
        print(f"Timeout testing image {image_path}")
        return None, None, None
    except Exception as e:
        print(f"Error testing image {image_path}: {e}")
        return None, None, None


def test_api_ocr(api_url, image_path, timeout=60, ocr_plate_text_mode='strict'):
    """
    Test OCR on API endpoint using the v2 process payload format.

    Args:
        api_url: Base URL of the API (e.g., http://localhost:9003)
        image_path: Path to image file
        timeout: Request timeout in seconds

    Returns:
        dict: API response with OCR results or None if error
    """
    try:
        # Resolve absolute path
        abs_image_path = os.path.abspath(image_path)

        # Prepare the exact payload format requested
        payload = {
            "processing_config": {
                "confidence_threshold": 0.3,
                "frames_per_second": 10,
                "ocr_confidence_threshold": 0.5,
                "ocr_match_confidence": 0.85,
                "ocr_plate_text_mode": ocr_plate_text_mode,
                "global_id_match_score": 0.7,
                "nms_threshold": 0,
                "similarity_threshold": 0,
                "spatial_threshold": 0,
                "max_disappeared": 0,
                "confirmation_frames": 0,
                "save_cropped_faces": True,
                "generate_embeddings": True,
                "embedding_model": "string",
                "embedding_detector_backend": "string",
                "min_face_size": 0,
                "enable_face_alignment": True,
                "custom_output_fps": 0,
                "enable_cross_camera_reid": True,
                "cross_camera_threshold": 0,
                "platform": "string",
                "extra": {
                    "additionalProp1": {}
                }
            },
            "inputs": [
                {
                    "id": str(uuid.uuid4()),
                    "input_type": "image_file",
                    "options": {
                        "uri": abs_image_path,
                        "camera_id": "string",
                        "lat": -90,
                        "lon": -180,
                        "pixels_per_meter": 10
                    },
                    "metadata": {
                        "additionalProp1": {}
                    }
                }
            ]
        }

        # We only need to hit the specific endpoint
        url = urljoin(api_url, '/v2/anpr_object/process')
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": str(uuid.uuid4())
        }

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error ({response.status_code}): {response.text}")
            return None

    except Exception as e:
        print(f"Error testing image {image_path}: {e}")
        return None


def extract_plate_from_response(response):
    """Extract plate text from ANPR v2 API response."""
    if not response or 'results' not in response:
        return None

    try:
        # The response format for v2: {"results": [{"detections": [{"ocr_text": "..."}]}]}
        for result in response.get('results', []):
            for detection in result.get('detections', []):
                plate_text = detection.get('ocr_text')
                if plate_text:
                    return str(plate_text).strip().upper()
    except Exception:
        pass

    return None


def calculate_accuracy(predicted, ground_truth):
    """Calculate character-level accuracy."""
    if not predicted or not ground_truth:
        return 0.0

    predicted = str(predicted).upper()
    ground_truth = str(ground_truth).upper()

    if predicted == ground_truth:
        return 1.0

    # Calculate character-level similarity
    matches = sum(1 for a, b in zip(predicted, ground_truth) if a == b)
    max_len = max(len(predicted), len(ground_truth))

    return matches / max_len if max_len > 0 else 0.0


async def process_image_task(session, api_url, image_path, idx, total, semaphore, ocr_plate_text_mode='strict'):
    """Process a single image with concurrency control."""
    async with semaphore:
        # Extract ground truth label
        ground_truth = extract_label_from_xml(image_path)
        if not ground_truth:
            ground_truth = extract_label_from_filename(image_path.name)

        if not ground_truth:
            print(f"[{idx}/{total}] ⚠ Skipping {image_path.name} - No valid label found")
            return {
                'image': image_path.name,
                'ground_truth': None,
                'predicted': None,
                'accuracy': 0.0,
                'valid': False,
                'request_time': 0.0,
                'request_id': None,
                'status': 'skipped',
                'reason': 'No valid label found in XML or filename'
            }

        # Test API
        print(f"[{idx}/{total}] Testing {image_path.name}...", end=" ")
        sys.stdout.flush()

        request_start = time.time()
        response, request_id, payload = await test_api_ocr_async(
            session, api_url, str(image_path), ocr_plate_text_mode=ocr_plate_text_mode
        )
        request_time = time.time() - request_start

        if response is None:
            print("❌ API Error")
            return {
                'image': image_path.name,
                'ground_truth': ground_truth,
                'predicted': None,
                'accuracy': 0.0,
                'valid': False,
                'request_time': request_time,
                'request_id': request_id,
                'status': 'error',
                'reason': 'API returned error or timeout'
            }

        # Extract prediction
        predicted = extract_plate_from_response(response)

        if not predicted:
            print("❌ No prediction")
            return {
                'image': image_path.name,
                'ground_truth': ground_truth,
                'predicted': None,
                'accuracy': 0.0,
                'valid': False,
                'request_time': request_time,
                'request_id': request_id,
                'status': 'no_prediction',
                'reason': 'No ocr_text found in API response detections'
            }

        # Calculate accuracy
        accuracy = calculate_accuracy(predicted, ground_truth)
        is_valid = is_valid_indian_plate(predicted)

        status = "✓" if accuracy == 1.0 else "~" if accuracy > 0.5 else "✗"
        print(f"{status} {predicted} (Accuracy: {accuracy * 100:.1f}%, Time: {request_time:.3f}s)")

        return {
            'image': image_path.name,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'accuracy': accuracy,
            'valid': is_valid,
            'request_time': request_time,
            'request_id': request_id,
            'status': 'success',
            'reason': None
        }


async def run_benchmark_async(api_url, dataset_path, max_images=None, concurrency=1, ocr_plate_text_mode='strict'):
    """Run OCR benchmark on dataset with concurrent requests."""

    # Find images
    image_files = find_image_files(dataset_path)

    if not image_files:
        print(f"No image files found in {dataset_path}")
        return None

    if max_images:
        image_files = image_files[:max_images]

    print(f"\nFound {len(image_files)} image(s) to test")
    print(f"API URL: {api_url}")
    print(f"Concurrency Level: {concurrency} concurrent user(s)")
    print(f"OCR Plate Text Mode: {ocr_plate_text_mode}")
    print(f"Testing OCR extraction performance...\n")

    start_time = time.time()

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Create tasks for all images
        tasks = [
            process_image_task(session, api_url, image_path, idx, len(image_files), semaphore, ocr_plate_text_mode)
            for idx, image_path in enumerate(image_files, 1)
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

    # Filter out None results (skipped images)
    results = [r for r in results if r is not None]

    total_time = time.time() - start_time

    return {
        'results': results,
        'total_time': total_time,
        'concurrency': concurrency,
        'ocr_plate_text_mode': ocr_plate_text_mode,
        'api_url': api_url,
        'dataset_path': str(dataset_path),
        'timestamp': datetime.now().isoformat()
    }


def run_benchmark(api_url, dataset_path, max_images=None, concurrency=1, ocr_plate_text_mode='strict'):
    """Run OCR benchmark on dataset (wrapper for async version)."""
    return asyncio.run(run_benchmark_async(api_url, dataset_path, max_images, concurrency, ocr_plate_text_mode))


def print_summary(benchmark_data):
    """Print benchmark summary."""
    results = benchmark_data['results']

    if not results:
        print("No results to summarize")
        return

    successful = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    no_pred = [r for r in results if r['status'] == 'no_prediction']
    concurrency = benchmark_data.get('concurrency', 1)
    ocr_plate_text_mode = benchmark_data.get('ocr_plate_text_mode', 'strict')

    print("\n" + "=" * 100)
    print("OCR TEXT EXTRACTION BENCHMARK SUMMARY")
    print("=" * 100 + "\n")

    print(f"Concurrency Level: {concurrency} concurrent user(s)")
    print(f"OCR Plate Text Mode: {ocr_plate_text_mode}")
    print(f"Total Images Tested: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)")
    print(f"Errors: {len(errors)} ({len(errors) / len(results) * 100:.1f}%)")
    print(f"No Prediction: {len(no_pred)} ({len(no_pred) / len(results) * 100:.1f}%)")

    if successful:
        accuracies = [r['accuracy'] for r in successful]
        request_times = [r['request_time'] for r in successful]
        valid_count = sum(1 for r in successful if r['valid'])

        print(f"\nAccuracy Metrics (Successful predictions):")
        print(
            f"  Perfect Match (100%): {sum(1 for a in accuracies if a == 1.0)} ({sum(1 for a in accuracies if a == 1.0) / len(accuracies) * 100:.1f}%)")
        print(
            f"  High Accuracy (>80%): {sum(1 for a in accuracies if a > 0.8)} ({sum(1 for a in accuracies if a > 0.8) / len(accuracies) * 100:.1f}%)")
        print(f"  Average Accuracy: {mean(accuracies) * 100:.2f}%")
        print(f"  Min Accuracy: {min(accuracies) * 100:.2f}%")
        print(f"  Max Accuracy: {max(accuracies) * 100:.2f}%")

        if len(accuracies) > 1:
            print(f"  Std Dev: {stdev(accuracies) * 100:.2f}%")

        print(f"\nValid Indian Plates: {valid_count} ({valid_count / len(successful) * 100:.1f}%)")

        print(f"\nPerformance Metrics:")
        print(f"  Avg Response Time: {mean(request_times):.3f}s")
        print(f"  Min Response Time: {min(request_times):.3f}s")
        print(f"  Max Response Time: {max(request_times):.3f}s")
        print(f"  Total Time: {benchmark_data['total_time']:.2f}s")
        print(f"  Throughput: {len(results) / benchmark_data['total_time']:.2f} images/sec")
        if concurrency > 1:
            print(f"  Effective Throughput: {len(successful) / benchmark_data['total_time']:.2f} successful/sec")

        # Top predictions
        print(f"\nTop Predicted Plates:")
        pred_counts = Counter(r['predicted'] for r in successful if r['predicted'])
        for plate, count in pred_counts.most_common(10):
            print(f"  {plate}: {count} detections")

    if no_pred:
        print(f"\nFiles with no OCR text:")
        for result in no_pred:
            print(f"  {result['image']} -> expected {result['ground_truth']}")

    print("\n" + "=" * 100)


def save_markdown_report(benchmark_data, output_file):
    """Save benchmark results as markdown report."""
    results = benchmark_data['results']
    successful = [r for r in results if r['status'] == 'success']
    no_pred = [r for r in results if r['status'] == 'no_prediction']

    with open(output_file, 'w') as f:
        f.write("# OCR Text Extraction API Benchmark Report\n\n")
        f.write(f"**Date:** {benchmark_data['timestamp']}\n\n")
        f.write(f"**API URL:** {benchmark_data['api_url']}\n\n")
        f.write(f"**Dataset:** {benchmark_data['dataset_path']}\n\n")
        f.write(f"**OCR Plate Text Mode:** {benchmark_data.get('ocr_plate_text_mode', 'strict')}\n\n")

        # Summary section
        concurrency = benchmark_data.get('concurrency', 1)
        f.write("## Summary\n\n")
        f.write(f"- **Concurrency Level:** {concurrency} concurrent user(s)\n")
        f.write(f"- **Total Images Tested:** {len(results)}\n")
        if len(results) > 0:
            f.write(f"- **Successful:** {len(successful)} ({len(successful) / len(results) * 100:.1f}%)\n")
        else:
            f.write(f"- **Successful:** 0 (0.0%)\n")
        f.write(f"- **Total Time:** {benchmark_data['total_time']:.2f}s\n")
        f.write(f"- **Throughput:** {len(results) / benchmark_data['total_time']:.2f} images/sec\n\n")

        if successful:
            accuracies = [r['accuracy'] for r in successful]
            request_times = [r['request_time'] for r in successful]
            valid_count = sum(1 for r in successful if r['valid'])

            f.write("## Accuracy Metrics\n\n")
            f.write(f"- **Average Accuracy:** {mean(accuracies) * 100:.2f}%\n")
            f.write(
                f"- **Perfect Match (100%):** {sum(1 for a in accuracies if a == 1.0)} ({sum(1 for a in accuracies if a == 1.0) / len(accuracies) * 100:.1f}%)\n")
            f.write(
                f"- **High Accuracy (>80%):** {sum(1 for a in accuracies if a > 0.8)} ({sum(1 for a in accuracies if a > 0.8) / len(accuracies) * 100:.1f}%)\n")
            f.write(f"- **Valid Indian Plates:** {valid_count} ({valid_count / len(successful) * 100:.1f}%)\n\n")

            f.write("## Performance Metrics\n\n")
            f.write(f"- **Avg Response Time:** {mean(request_times):.3f}s\n")
            f.write(f"- **Min Response Time:** {min(request_times):.3f}s\n")
            f.write(f"- **Max Response Time:** {max(request_times):.3f}s\n")
            if concurrency > 1:
                f.write(
                    f"- **Effective Throughput:** {len(successful) / benchmark_data['total_time']:.2f} successful/sec\n")
            f.write("\n")

            # Detailed results table
            f.write("## Detailed Results\n\n")
            f.write("| Image | Ground Truth | Predicted | Accuracy | Valid | Time (s) |\n")
            f.write("|-------|--------------|-----------|----------|-------|----------|\n")

            for result in successful:
                accuracy_pct = result['accuracy'] * 100
                valid_mark = "✓" if result['valid'] else "✗"
                f.write(
                    f"| {result['image']} | {result['ground_truth']} | {result['predicted']} | {accuracy_pct:.1f}% | {valid_mark} | {result['request_time']:.3f} |\n")

        if no_pred:
            f.write("\n## Files With No OCR Text\n\n")
            f.write("| Image | Ground Truth | Time (s) |\n")
            f.write("|-------|--------------|----------|\n")
            for result in no_pred:
                f.write(f"| {result['image']} | {result['ground_truth']} | {result['request_time']:.3f} |\n")

        f.write("\n---\n")
        f.write(f"*Report generated on {benchmark_data['timestamp']}*\n")

    print(f"\n✓ Markdown report saved: {output_file}")


def save_json_report(benchmark_data, output_file):
    """Save benchmark results as JSON."""
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"✓ JSON report saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='OCR Text Extraction API Benchmark')
    parser.add_argument('--api-url', default='http://localhost:9003',
                        help='API URL (default: http://localhost:9003)')
    parser.add_argument('--dataset-path', default='../Indian vehicle license plate dataset/video_images',
                        help='Path to dataset directory')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to test')
    parser.add_argument('--concurrency', type=int, default=3,
                        help='Number of concurrent users (1-3, default: 3)')
    parser.add_argument('--json', action='store_true',
                        help='Save JSON report')
    parser.add_argument('--markdown', action='store_true',
                        help='Save markdown report')
    parser.add_argument('--output-dir', default='./benchmark',
                        help='Output directory for reports')
    parser.add_argument('--ocr-plate-text-mode', default='strict', choices=['balanced', 'strict'],
                        help='OCR plate text mode to send in processing_config')
    parser.add_argument('report_files', nargs='*', help='Two JSON report files to compare')

    args = parser.parse_args()

    # Comparison mode: if two JSON files are provided, compare and exit
    if len(getattr(args, 'report_files', [])) == 2:
        p1 = Path(args.report_files[0])
        p2 = Path(args.report_files[1])
        if not p1.exists() or not p2.exists():
            print(f"Error: file not found: {p1 if not p1.exists() else p2}")
            sys.exit(2)
        m1 = load_metrics(p1)
        m2 = load_metrics(p2)
        print("\nOCR Benchmark Comparison (console table)\n")
        print(format_table([m1, m2]))
        print("\nDeltas (second - first):")
        interesting = [
            ("success", "Success"),
            ("no_pred", "No Pred"),
            ("avg_accuracy_pct", "Avg Acc %"),
            ("perfect", "Perfect"),
            ("gt80", ">80%"),
            ("gt60", ">60%"),
            ("gt50", ">50%"),
            ("avg_time_s", "Avg Time (s)"),
            ("throughput_img_per_s", "Throughput (img/s)"),
        ]
        for key, name in interesting:
            a = m1.get(key, 0)
            b = m2.get(key, 0)
            try:
                delta = round(float(b) - float(a), 2)
            except Exception:
                delta = "-"
            print(f"- {name}: {a} -> {b} (Δ {delta})")
        print("\nTip: copy results into Excel by pasting this table, or run through `column -t` for monospace alignment.")
        return

    # Validate concurrency
    if args.concurrency < 1 or args.concurrency > 3:
        print("Error: Concurrency must be between 1 and 3")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    benchmark_data = run_benchmark(
        args.api_url,
        args.dataset_path,
        args.max_images,
        args.concurrency,
        args.ocr_plate_text_mode,
    )

    if not benchmark_data:
        print("Benchmark failed")
        sys.exit(1)

    # Print summary
    print_summary(benchmark_data)

    # Save reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = benchmark_data.get('ocr_plate_text_mode', args.ocr_plate_text_mode)

    if args.markdown or True:  # Always save markdown by default
        md_file = output_dir / f'api_ocr_benchmark_{timestamp}_{mode_suffix}.md'
        save_markdown_report(benchmark_data, str(md_file))

    if args.json:
        json_file = output_dir / f'api_ocr_benchmark_{timestamp}_{mode_suffix}.json'
        save_json_report(benchmark_data, str(json_file))


if __name__ == '__main__':
    main()
