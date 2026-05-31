# API OCR Text Extraction Benchmark (Quick Commands)

## What this is
Copy-paste ready commands to run the OCR API benchmark against the Indian vehicle license plate dataset, with examples for balanced and strict modes, concurrency, and JSON/Markdown outputs. A small Python tool is also provided to compare two JSON reports in the console (Excel-like table).

## Quick reference

Run google_images with STRICT mode, concurrency=3:
```bash
uv run python benchmark/api_ocr_benchmark.py \
  --api-url http://localhost:9003 \
  --dataset-path "/home/harsha/abhishek/vlm-video-captioning/ANPR/anpr-inference-service/Indian vehicle license plate dataset/google_images" \
  --ocr-plate-text-mode strict \
  --concurrency 3 \
  --json
```

Run google_images with BALANCED mode, concurrency=3:
```bash
uv run python benchmark/api_ocr_benchmark.py \
  --api-url http://localhost:9003 \
  --dataset-path "/home/harsha/abhishek/vlm-video-captioning/ANPR/anpr-inference-service/Indian vehicle license plate dataset/google_images" \
  --ocr-plate-text-mode balanced \
  --concurrency 3 \
  --json
```

Compare two JSON reports:
```bash
python3 benchmark/compare_benchmark_json.py \
  benchmark/api_ocr_benchmark_20260403_152035_balanced.json \
  benchmark/api_ocr_benchmark_20260403_154846_balanced.json 
  # benchmark/api_ocr_benchmark_20260403_152007_strict.json
```

Status
- README updated with strict/balanced concurrent commands and comparison workflow.
- New comparison utility created and ready to use.

## Prerequisites
- ANPR API running at `http://localhost:9003`
- Dataset path (examples below use google_images):
  `/home/harsha/abhishek/vlm-video-captioning/ANPR/anpr-inference-service/Indian vehicle license plate dataset/google_images`

Check API:
```bash
curl http://localhost:9003/health
```

## Run: google_images with STRICT mode, concurrency=3
```bash
uv run python benchmark/api_ocr_benchmark.py \
  --api-url http://localhost:9003 \
  --dataset-path "/home/harsha/abhishek/vlm-video-captioning/ANPR/anpr-inference-service/Indian vehicle license plate dataset/google_images" \
  --ocr-plate-text-mode strict \
  --concurrency 3 \
  --json
```

## Run: google_images with BALANCED mode, concurrency=3
```bash
uv run python benchmark/api_ocr_benchmark.py \
  --api-url http://localhost:9003 \
  --dataset-path "/home/harsha/abhishek/vlm-video-captioning/ANPR/anpr-inference-service/Indian vehicle license plate dataset/google_images" \
  --ocr-plate-text-mode balanced \
  --concurrency 3 \
  --json
```

## Other common scenarios

- Limit images to 100:
```bash
uv run python benchmark/api_ocr_benchmark.py \
  --api-url http://localhost:9003 \
  --dataset-path "/path/to/images" \
  --ocr-plate-text-mode balanced \
  --concurrency 3 \
  --max-images 100 \
  --json
```

- Change output directory for reports:
```bash
uv run python benchmark/api_ocr_benchmark.py \
  --api-url http://localhost:9003 \
  --dataset-path "/path/to/images" \
  --ocr-plate-text-mode strict \
  --concurrency 3 \
  --output-dir benchmark \
  --json
```

## Where reports are saved
Markdown and JSON reports are saved into the `benchmark/` folder as `api_ocr_benchmark_*.md` and `api_ocr_benchmark_*.json`.

## Console comparison (Excel-like) between two JSON reports
Use the helper script `benchmark/compare_benchmark_json.py` to compare two benchmark JSONs (e.g., balanced vs strict) and print a compact table in the console.

Example:
```bash
python3 benchmark/compare_benchmark_json.py \
  benchmark/api_ocr_benchmark_20260331_172512_balanced.json \
  benchmark/api_ocr_benchmark_20260331_172718_strict.json
```

Output includes for each report:
- Mode, Total, Success, No Pred, Errors
- Avg Acc %, Perfect, >80%
- Avg Time (s), Total Time (s), Throughput (img/s)
and a quick delta summary.

## Full CLI reference
```bash
python3 benchmark/api_ocr_benchmark.py [OPTIONS]

  --api-url URL                 API endpoint (default: http://localhost:9003)
  --dataset-path PATH           Path to labeled images
  --ocr-plate-text-mode MODE    balanced | strict (default: balanced)
  --concurrency N               Concurrent requests (default: 3)
  --max-images N                Limit number of images to test
  --json                        Save JSON report
  --markdown                    Save Markdown report (default true)
  --output-dir DIR              Report directory (default: ./benchmark)
```

