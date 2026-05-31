import os
import re

os.environ.setdefault("APP_NAME", "ANPR TEST SERVICE")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("TRITON_SERVER_URL", "127.0.0.1:9001")
os.environ.setdefault("TRITON_PROTOCOL", "grpc")
os.environ.setdefault("VEHICLE_MODEL_NAME", "vehicle_detection_rt_detr")
os.environ.setdefault("VEHICLE_CLASS_ID_MAP", '{"0":"vehicle"}')
os.environ.setdefault("PLATE_MODEL_NAME", "plate_region_detection_rt_detr")
os.environ.setdefault("PLATE_CLASS_ID_MAP", '{"0":"number_plate"}')
os.environ.setdefault("PLATE_CANDIDATE_VEHICLE_CLASSES", "vehicle,car,bus,truck")
os.environ.setdefault("BATCH_SIZE", "4")
os.environ.setdefault("INPUT_FOLDER", "data/input")
os.environ.setdefault("OUTPUT_FOLDER", "data/output")
os.environ.setdefault("IOU_THRESHOLD", "0.5")
os.environ.setdefault("BLUR_THRESHOLD", "40.0")
os.environ.setdefault("PLATE_BLUR_THRESHOLD", "60.0")
os.environ.setdefault("VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.mkv,.webm")
os.environ.setdefault("IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.bmp,.JPG,.JPEG,.PNG")
os.environ.setdefault("OCR_TIMEOUT", "3")
os.environ.setdefault("OCR_LANGUAGES", "en")
os.environ.setdefault("X_PADDING", "0.20")
os.environ.setdefault("Y_PADDING", "0.20")
os.environ.setdefault(
    "OCR_CLEANUP_REGEX",
    r"(?ix)^(?:[A-Z]{2}\d{1,2}[A-Z]?[A-Z]{1,3}\d{4}[A-Z]?|\d{2}BH\d{4}[A-Z]{2}|\d{2}(?:CD|CC|UN)\d{1,4})$",
)
os.environ.setdefault("OCR_DET_LIMIT_SIDE_LEN", "736")
os.environ.setdefault("OCR_DET_BOX_THRESH", "0.7")
os.environ.setdefault("OCR_REC_SCORE_THRESH", "0.7")
os.environ.setdefault("OCR_DET_THRESH", "0.25")
os.environ.setdefault("OCR_DET_UNCLIP_RATIO", "1.8")
os.environ.setdefault("OCR_ENABLE_PREPROCESSING", "true")
os.environ.setdefault("OCR_CLAHE_CLIP_LIMIT", "2.0")
os.environ.setdefault("OCR_DENOISE_STRENGTH", "10")
os.environ.setdefault("GLOBAL_TRACK_LOOKBACK_SECONDS", "300")
os.environ.setdefault("GLOBAL_TRACK_DATABASE_URL", "sqlite:///./global_tracking.db")

from triton_server.app import settings
from triton_server.app import OCRUtils

settings.app_name = os.environ["APP_NAME"]
settings.app_env = os.environ["APP_ENV"]
settings.log_level = os.environ["LOG_LEVEL"]
settings.triton_server_url = os.environ["TRITON_SERVER_URL"]
settings.triton_protocol = os.environ["TRITON_PROTOCOL"]
settings.vehicle_model_name = os.environ["VEHICLE_MODEL_NAME"]
settings.vehicle_class_id_map = os.environ["VEHICLE_CLASS_ID_MAP"]
settings.plate_model_name = os.environ["PLATE_MODEL_NAME"]
settings.plate_class_id_map = os.environ["PLATE_CLASS_ID_MAP"]
settings.plate_candidate_vehicle_classes = os.environ["PLATE_CANDIDATE_VEHICLE_CLASSES"]
settings.batch_size = int(os.environ["BATCH_SIZE"])
settings.input_folder = os.environ["INPUT_FOLDER"]
settings.output_folder = os.environ["OUTPUT_FOLDER"]
settings.iou_threshold = float(os.environ["IOU_THRESHOLD"])
settings.blur_threshold = float(os.environ["BLUR_THRESHOLD"])
settings.plate_blur_threshold = float(os.environ["PLATE_BLUR_THRESHOLD"])
settings.video_extensions = os.environ["VIDEO_EXTENSIONS"]
settings.image_extensions = os.environ["IMAGE_EXTENSIONS"]
settings.ocr_timeout = int(os.environ["OCR_TIMEOUT"])
settings.ocr_languages = os.environ["OCR_LANGUAGES"]
settings.x_padding = float(os.environ["X_PADDING"])
settings.y_padding = float(os.environ["Y_PADDING"])
settings.ocr_cleanup_regex = os.environ["OCR_CLEANUP_REGEX"]
settings.ocr_det_limit_side_len = int(os.environ["OCR_DET_LIMIT_SIDE_LEN"])
settings.ocr_det_box_thresh = float(os.environ["OCR_DET_BOX_THRESH"])
settings.ocr_rec_score_thresh = float(os.environ["OCR_REC_SCORE_THRESH"])
settings.ocr_det_thresh = float(os.environ["OCR_DET_THRESH"])
settings.ocr_det_unclip_ratio = float(os.environ["OCR_DET_UNCLIP_RATIO"])
settings.ocr_enable_preprocessing = os.environ["OCR_ENABLE_PREPROCESSING"].lower() == "true"
settings.ocr_clahe_clip_limit = float(os.environ["OCR_CLAHE_CLIP_LIMIT"])
settings.ocr_denoise_strength = int(os.environ["OCR_DENOISE_STRENGTH"])
settings.global_track_lookback_seconds = int(os.environ["GLOBAL_TRACK_LOOKBACK_SECONDS"])
settings.global_track_database_url = os.environ["GLOBAL_TRACK_DATABASE_URL"]

OCRUtils.PLATE_PATTERN = re.compile(settings.ocr_cleanup_regex)
