import cv2
import numpy as np
import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import pickle
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
import uuid

class KalmanTracker:
    """Kalman filter for tracking face bounding box centers with velocity."""
    
    def __init__(self, initial_bbox):
        """Initialize Kalman filter with initial bounding box."""
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, vx, vy), 2 measurement vars (x, y)
        
        # Transition matrix (constant velocity model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x' = x + vx
            [0, 1, 0, 1],  # y' = y + vy
            [0, 0, 1, 0],  # vx' = vx
            [0, 0, 0, 1]   # vy' = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we observe x, y)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = 5.0 * np.eye(2, dtype=np.float32)
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initialize state with bbox center and zero velocity
        center_x = (initial_bbox[0] + initial_bbox[2]) / 2
        center_y = (initial_bbox[1] + initial_bbox[3]) / 2
        self.kalman.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)
        
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        
    def predict(self):
        """Predict next position."""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        predicted = self.kalman.predict()
        return predicted[:2]  # return only x, y
    
    def update(self, bbox):
        """Update filter with new bounding box."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        measurement = np.array([center_x, center_y], dtype=np.float32)
        
        self.kalman.correct(measurement)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
    def get_state(self):
        """Get current state (center position)."""
        return self.kalman.statePost[:2]

class RobustFaceTracker:
    """
    Enhanced face tracking system with motion prediction, temporal consistency,
    and robust feature matching for stable face ID assignment.
    MINIMAL UUID CHANGE: Only ID generation changed, all tracking logic preserved.
    """
    
    def __init__(self, db_path: str = None, 
                 similarity_threshold: float = 0.65,
                 spatial_threshold: float = 200.0,
                 max_disappeared: int = 50,
                 confirmation_frames: int = 3,
                 output_base_dir: str = "output",
                 save_cropped_faces: bool = True):
        """
        Initialize robust face tracker.
        
        Args:
            db_path: Database file path (None to disable database)
            similarity_threshold: Face similarity threshold (higher = more strict)
            spatial_threshold: Maximum distance for same face between frames
            max_disappeared: Maximum frames before considering track lost
            confirmation_frames: Frames needed to confirm new face
            save_cropped_faces: Whether to save person-specific cropped face images
        """
        self.db_path = db_path
        self.use_database = db_path is not None
        self.similarity_threshold = similarity_threshold
        self.spatial_threshold = spatial_threshold
        self.max_disappeared = max_disappeared
        self.confirmation_frames = confirmation_frames
        self.output_base_dir = output_base_dir
        self.save_cropped_faces = save_cropped_faces
        
        # Remove sequential ID counter - using UUIDs now
        self.active_tracks = {}  # tracking_id -> track_info
        self.kalman_trackers = {}  # tracking_id -> KalmanTracker
        self.face_embeddings = {}  # tracking_id -> list of embeddings (for temporal averaging)
        self.face_embeddings_buffer = {}  # tracking_id -> deque of recent embeddings
        self.candidate_faces = {}  # temp storage for faces awaiting confirmation
        self.frame_count = 0
        
        # Temporal smoothing
        self.embedding_buffer_size = 10
        self.decision_history = defaultdict(lambda: deque(maxlen=5))  # track decision history
        
        # Database operations disabled - no longer saving to SQLite
        # self._init_database()
        # self._load_existing_faces()
        
        # Track face detections for JSON export
        self.face_detections_log = []  # Store all detections for JSON export
        
        print(f"RobustFaceTracker initialized with {len(self.face_embeddings)} existing faces")

    def _generate_unique_id(self) -> str:
        """Generate a globally unique ID for face tracking."""
        # Generate UUID4 and convert to short format for readability
        full_uuid = str(uuid.uuid4())
        # Use first 8 characters for shorter, readable IDs
        short_id = full_uuid[:8].upper()
        return f"{short_id}"
    
    def _init_database(self):
        """Initialize SQLite database for face storage with migration support."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist and their structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='unique_faces'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Check current schema
            cursor.execute("PRAGMA table_info(unique_faces)")
            columns = cursor.fetchall()
            tracking_id_column = next((col for col in columns if col[1] == 'tracking_id'), None)
            
            if tracking_id_column and tracking_id_column[2] == 'INTEGER':
                # Need to migrate from INTEGER to TEXT
                print("🔄 Migrating database from INTEGER to UUID tracking IDs...")
                self._migrate_to_uuid_schema(cursor)
            else:
                # Migrate database schema if needed
                self._migrate_database_schema(cursor)
        else:
            # Create base tables if they don't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unique_faces (
                    tracking_id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    appearance_count INTEGER DEFAULT 1,
                    embedding BLOB,
                    sample_image_path TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_appearances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracking_id TEXT,
                    frame_number INTEGER,
                    bbox_x REAL,
                    bbox_y REAL,
                    bbox_w REAL,
                    bbox_h REAL,
                    confidence REAL,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (tracking_id) REFERENCES unique_faces (tracking_id)
                )
            ''')
            
            # Migrate database schema if needed
            self._migrate_database_schema(cursor)
        
        conn.commit()
        conn.close()

    def _migrate_to_uuid_schema(self, cursor):
        """Migrate existing INTEGER tracking_id database to UUID TEXT schema."""
        try:
            # Create new tables with UUID schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unique_faces_uuid (
                    tracking_id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    appearance_count INTEGER DEFAULT 1,
                    embedding BLOB,
                    sample_image_path TEXT,
                    metadata TEXT,
                    confidence_avg REAL,
                    bbox_width_avg REAL,
                    bbox_height_avg REAL,
                    original_id INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_appearances_uuid (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracking_id TEXT,
                    frame_number INTEGER,
                    bbox_x REAL,
                    bbox_y REAL,
                    bbox_w REAL,
                    bbox_h REAL,
                    confidence REAL,
                    timestamp TIMESTAMP,
                    predicted_x REAL,
                    predicted_y REAL,
                    match_distance REAL,
                    original_tracking_id INTEGER,
                    FOREIGN KEY (tracking_id) REFERENCES unique_faces_uuid (tracking_id)
                )
            ''')
            
            # Migrate data from old tables to new tables
            cursor.execute("SELECT * FROM unique_faces")
            old_faces = cursor.fetchall()
            
            id_mapping = {}  # old_id -> new_uuid
            
            for row in old_faces:
                old_id = row[0]  # INTEGER tracking_id
                new_uuid = self._generate_unique_id()
                id_mapping[old_id] = new_uuid
                
                # Handle variable row lengths
                row_data = list(row[1:])  # Skip old tracking_id
                while len(row_data) < 9:  # Ensure we have enough columns
                    row_data.append(None)
                
                # Insert into new table with UUID
                cursor.execute('''
                    INSERT INTO unique_faces_uuid 
                    (tracking_id, first_seen, last_seen, appearance_count, embedding, 
                     sample_image_path, metadata, confidence_avg, bbox_width_avg, bbox_height_avg, original_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (new_uuid, *row_data[:6], old_id))
            
            # Migrate face_appearances if they exist
            try:
                cursor.execute("SELECT * FROM face_appearances")
                old_appearances = cursor.fetchall()
                
                for row in old_appearances:
                    old_tracking_id = row[1]  # INTEGER tracking_id from appearances
                    new_uuid = id_mapping.get(old_tracking_id)
                    if new_uuid:
                        # Handle variable row lengths
                        appearance_data = list(row[2:])  # Skip id and tracking_id
                        while len(appearance_data) < 9:  # Ensure enough columns
                            appearance_data.append(None)
                        
                        cursor.execute('''
                            INSERT INTO face_appearances_uuid 
                            (tracking_id, frame_number, bbox_x, bbox_y, bbox_w, bbox_h, 
                             confidence, timestamp, predicted_x, predicted_y, match_distance, original_tracking_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (new_uuid, *appearance_data[:6], None, None, None, old_tracking_id))
            except sqlite3.Error:
                pass  # face_appearances might not exist yet
            
            # Backup old tables and replace with new ones
            cursor.execute("ALTER TABLE unique_faces RENAME TO unique_faces_backup")
            try:
                cursor.execute("ALTER TABLE face_appearances RENAME TO face_appearances_backup")
            except sqlite3.Error:
                pass  # Table might not exist
            cursor.execute("ALTER TABLE unique_faces_uuid RENAME TO unique_faces")
            cursor.execute("ALTER TABLE face_appearances_uuid RENAME TO face_appearances")
            
            print(f"✅ Successfully migrated {len(old_faces)} faces to UUID format")
            print("📝 Original tables backed up as unique_faces_backup and face_appearances_backup")
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            # Rollback by dropping UUID tables if they exist
            cursor.execute("DROP TABLE IF EXISTS unique_faces_uuid")
            cursor.execute("DROP TABLE IF EXISTS face_appearances_uuid")
            raise
    
    def _migrate_database_schema(self, cursor):
        """Migrate database schema to add new columns if they don't exist."""
        # Check and add new columns to unique_faces table
        cursor.execute("PRAGMA table_info(unique_faces)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        
        new_columns = [
            ('confidence_avg', 'REAL'),
            ('bbox_width_avg', 'REAL'), 
            ('bbox_height_avg', 'REAL')
        ]
        
        for column_name, column_type in new_columns:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f'ALTER TABLE unique_faces ADD COLUMN {column_name} {column_type}')
                    print(f"✅ Added column '{column_name}' to unique_faces table")
                except sqlite3.Error as e:
                    print(f"⚠️  Could not add column '{column_name}': {e}")
        
        # Check and add new columns to face_appearances table  
        cursor.execute("PRAGMA table_info(face_appearances)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        
        appearance_columns = [
            ('predicted_x', 'REAL'),
            ('predicted_y', 'REAL'),
            ('match_distance', 'REAL')
        ]
        
        for column_name, column_type in appearance_columns:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f'ALTER TABLE face_appearances ADD COLUMN {column_name} {column_type}')
                    print(f"✅ Added column '{column_name}' to face_appearances table")
                except sqlite3.Error as e:
                    print(f"⚠️  Could not add column '{column_name}': {e}")
    
    def _load_existing_faces(self):
        """Load existing face embeddings from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT tracking_id, embedding FROM unique_faces")
            for tracking_id, embedding_blob in cursor.fetchall():
                if embedding_blob:
                    try:
                        embedding_list = pickle.loads(embedding_blob)
                        # Convert old single embeddings to list format
                        if isinstance(embedding_list, np.ndarray):
                            embedding_list = [embedding_list]
                        self.face_embeddings[tracking_id] = embedding_list
                        self.face_embeddings_buffer[tracking_id] = deque(
                            embedding_list[-self.embedding_buffer_size:], 
                            maxlen=self.embedding_buffer_size
                        )
                    except Exception as e:
                        print(f"Warning: Could not load embedding for face {tracking_id}: {e}")
        except sqlite3.Error as e:
            print(f"Warning: Database error while loading faces: {e}")
        
        conn.close()
    
    def _extract_enhanced_face_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract enhanced face embedding with better pose and lighting invariance.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512, dtype=np.float32)
        
        face_roi = frame[y1:y2, x1:x2]
        
        # Multi-scale feature extraction
        features = []
        
        # Scale 1: Full face (128x128)
        face_128 = cv2.resize(face_roi, (128, 128))
        features.append(self._extract_scale_features(face_128, 160))
        
        # Scale 2: Face center (focusing on nose/mouth area)
        h, w = face_roi.shape[:2]
        center_crop = face_roi[int(h*0.125):int(h*0.875), int(w*0.125):int(w*0.875)]
        if center_crop.size > 0:
            face_center = cv2.resize(center_crop, (128, 128))
            features.append(self._extract_scale_features(face_center, 160))
        else:
            features.append(np.zeros(160))
        
        # Scale 3: Eye region (most distinctive features)
        eye_region = face_roi[int(h*0.2):int(h*0.6), int(w*0.1):int(w*0.9)]
        if eye_region.size > 0:
            eye_resized = cv2.resize(eye_region, (64, 32))
            features.append(self._extract_scale_features(eye_resized, 96))
        else:
            features.append(np.zeros(96))
        
        # Combine and normalize
        embedding = np.concatenate(features)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Ensure fixed size (512)
        if len(embedding) < 512:
            embedding = np.pad(embedding, (0, 512 - len(embedding)), 'constant')
        elif len(embedding) > 512:
            embedding = embedding[:512]
        
        return embedding.astype(np.float32)
    
    def _extract_scale_features(self, face_img: np.ndarray, target_size: int) -> np.ndarray:
        """Extract features from face image at specific scale."""
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_img
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_gray = clahe.apply(face_gray)
        
        features = []
        
        # 1. Enhanced histogram features
        hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)
        
        # 2. Enhanced LBP features (safe implementation)
        try:
            lbp_features = self._extract_enhanced_lbp(face_gray)
            features.append(lbp_features)
        except Exception as e:
            # Fallback to zeros if LBP fails
            features.append(np.zeros(24, dtype=np.float32))
        
        # 3. Gabor filter features (safe implementation)
        try:
            gabor_features = self._extract_gabor_features(face_gray)
            features.append(gabor_features)
        except Exception as e:
            # Fallback to zeros if Gabor fails
            features.append(np.zeros(16, dtype=np.float32))
        
        # 4. Statistical features
        try:
            stat_features = self._extract_statistical_moments(face_gray)
            features.append(stat_features)
        except Exception as e:
            # Fallback to zeros if statistical features fail
            features.append(np.zeros(5, dtype=np.float32))

        # Combine and resize to target size
        combined = np.concatenate(features)
        
        if len(combined) < target_size:
            combined = np.pad(combined, (0, target_size - len(combined)), 'constant')
        elif len(combined) > target_size:
            # Use PCA-like dimensionality reduction (simplified)
            step = len(combined) // target_size
            combined = combined[::step][:target_size]
        
        return combined
    
    def _extract_enhanced_lbp(self, face_gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Extract enhanced Local Binary Pattern features."""
        h, w = face_gray.shape
        # Use uint16 to handle larger values, then convert safely
        lbp = np.zeros((h, w), dtype=np.uint16)
        
        # Calculate LBP with reduced points to stay within reasonable range
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = face_gray[i, j]
                code = 0
                for k in range(n_points):
                    x = i + radius * np.cos(2 * np.pi * k / n_points)
                    y = j - radius * np.sin(2 * np.pi * k / n_points)
                    x, y = int(round(x)), int(round(y))
                    if 0 <= x < h and 0 <= y < w:
                        if face_gray[x, y] >= center:
                            code |= (1 << k)
                # Ensure code fits in valid range
                lbp[i, j] = min(code, 255)
        
        # Convert to uint8 safely
        lbp_uint8 = lbp.astype(np.uint8)
        
        # Create uniform LBP histogram
        hist, _ = np.histogram(lbp_uint8.ravel(), bins=32, range=(0, 256))
        hist = hist.astype(np.float32)
        hist = cv2.normalize(hist, hist).flatten()
        return hist[:24]  # Use first 24 bins
    
    def _extract_gabor_features(self, face_gray: np.ndarray) -> np.ndarray:
        """Extract Gabor filter features for texture analysis."""
        features = []
        
        # Multiple orientations and frequencies
        orientations = [0, 45, 90, 135]
        frequencies = [0.1, 0.3]
        
        for freq in frequencies:
            for angle in orientations:
                try:
                    kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 
                                              2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                    # Use correct output type
                    filtered = cv2.filter2D(face_gray, cv2.CV_32F, kernel)
                    # Ensure finite values
                    mean_val = np.mean(filtered) if np.isfinite(np.mean(filtered)) else 0.0
                    var_val = np.var(filtered) if np.isfinite(np.var(filtered)) else 0.0
                    features.extend([mean_val, var_val])
                except Exception:
                    features.extend([0.0, 0.0])
        
        # Ensure we have exactly 16 features
        while len(features) < 16:
            features.append(0.0)
            
        return np.array(features[:16], dtype=np.float32)
    
    def _extract_statistical_moments(self, face_gray: np.ndarray) -> np.ndarray:
        """Extract statistical moments of pixel intensities."""
        pixels = face_gray.flatten().astype(np.float32)
        
        mean_val = np.mean(pixels)
        std_val = np.std(pixels)
        skewness = np.mean(((pixels - mean_val) / std_val) ** 3) if std_val > 0 else 0
        kurtosis = np.mean(((pixels - mean_val) / std_val) ** 4) if std_val > 0 else 0
        
        # Edge density
        edges = cv2.Canny(face_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return np.array([mean_val/255, std_val/255, skewness, kurtosis, edge_density], dtype=np.float32)
    
    def _calculate_enhanced_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate enhanced similarity between embeddings."""
        # Cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Euclidean distance similarity
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combine similarities
        combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
        
        return max(0.0, min(1.0, combined_sim))
    
    def _find_best_matches(self, detections: List[Tuple[np.ndarray, np.ndarray, float]], 
                          frame_number: int) -> List[Tuple[int, str, float]]:
        """
        Find optimal assignment between detections and existing tracks using Hungarian algorithm.
        
        Returns:
            List of (detection_idx, track_id, similarity_score) tuples
        """
        if not detections or not self.active_tracks:
            return []
        
        # Predict positions for all active tracks
        predictions = {}
        for track_id in self.active_tracks.keys():
            if track_id in self.kalman_trackers:
                predicted_pos = self.kalman_trackers[track_id].predict()
                predictions[track_id] = predicted_pos
        
        # Calculate cost matrix (detection vs track)
        cost_matrix = np.full((len(detections), len(self.active_tracks)), 1.0)
        track_ids = list(self.active_tracks.keys())
        
        for det_idx, (bbox, embedding, confidence) in enumerate(detections):
            det_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            for track_idx, track_id in enumerate(track_ids):
                # Skip tracks that are too old
                if frame_number - self.active_tracks[track_id]['last_seen'] > self.max_disappeared // 2:
                    continue
                
                # Calculate similarity cost
                similarity_cost = 1.0
                if track_id in self.face_embeddings_buffer:
                    # Use average of recent embeddings
                    track_embeddings = list(self.face_embeddings_buffer[track_id])
                    if track_embeddings:
                        avg_embedding = np.mean(track_embeddings, axis=0)
                        similarity = self._calculate_enhanced_similarity(embedding, avg_embedding)
                        similarity_cost = 1.0 - similarity
                
                # Calculate spatial cost
                spatial_cost = 1.0
                if track_id in predictions:
                    predicted_center = predictions[track_id]
                    distance = np.linalg.norm(det_center - predicted_center)
                    spatial_cost = min(1.0, distance / self.spatial_threshold)
                
                # Temporal consistency cost
                temporal_cost = 0.0
                if track_id in self.decision_history:
                    recent_decisions = list(self.decision_history[track_id])
                    if recent_decisions:
                        # Favor tracks that have been consistently matched
                        consistency = sum(1 for d in recent_decisions if d == 'matched') / len(recent_decisions)
                        temporal_cost = 0.2 * (1.0 - consistency)
                
                # Combined cost (lower is better)
                total_cost = 0.4 * similarity_cost + 0.4 * spatial_cost + 0.2 * temporal_cost
                cost_matrix[det_idx, track_idx] = total_cost
        
        # Apply Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on thresholds
        matches = []
        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            track_id = track_ids[track_idx]
            
            # Convert cost back to similarity for thresholding
            similarity = 1.0 - cost
            
            # Accept match if similarity is above threshold
            if similarity > self.similarity_threshold:
                matches.append((det_idx, track_id, similarity))
        
        return matches

    def _create_new_track(self, bbox: np.ndarray, embedding: np.ndarray, 
                         confidence: float, frame_number: int, frame: np.ndarray) -> str:
        """Create new track for unmatched detection. ONLY CHANGE: UUID generation."""
        track_id = self._generate_unique_id()  # <-- ONLY CHANGE: UUID instead of sequential
        
        # Initialize Kalman tracker
        self.kalman_trackers[track_id] = KalmanTracker(bbox)
        
        # Create track info
        self.active_tracks[track_id] = {
            'first_seen': frame_number,
            'last_seen': frame_number,
            'bbox': bbox,
            'confidence': confidence,
            'appearance_count': 1,
            'confirmed': False  # New tracks need confirmation
        }
        
        # Initialize embedding buffer
        self.face_embeddings_buffer[track_id] = deque([embedding], maxlen=self.embedding_buffer_size)
        
        # Store in candidate faces for confirmation
        self.candidate_faces[track_id] = {
            'embeddings': [embedding],
            'frame_count': 1,
            'sample_frame': frame.copy(),
            'sample_bbox': bbox.copy()
        }
        
        return track_id

    def _save_confirmed_face(self, track_id: str, candidate: Dict):
        """Save confirmed face (database operations disabled)."""
        # Average embeddings
        avg_embedding = np.mean(candidate['embeddings'], axis=0)
        self.face_embeddings[track_id] = [avg_embedding]
        
        # Save sample image
        sample_path = self._save_face_sample(candidate['sample_frame'], 
                                           candidate['sample_bbox'], track_id)
        
        # Database operations disabled - no longer saving to SQLite
        # conn = sqlite3.connect(self.db_path)
        # ... (database code removed)

    def _update_track(self, track_id: str, bbox: np.ndarray, embedding: np.ndarray, 
                     confidence: float, frame_number: int):
        """Update existing track with new detection."""
        # Update Kalman filter
        if track_id not in self.kalman_trackers:
            self.kalman_trackers[track_id] = KalmanTracker(bbox)
        else:
            self.kalman_trackers[track_id].update(bbox)
        
        # Update track info
        self.active_tracks[track_id].update({
            'last_seen': frame_number,
            'bbox': bbox,
            'confidence': confidence,
            'appearance_count': self.active_tracks[track_id]['appearance_count'] + 1
        })
        
        # Update embedding buffer
        if track_id not in self.face_embeddings_buffer:
            self.face_embeddings_buffer[track_id] = deque(maxlen=self.embedding_buffer_size)
        self.face_embeddings_buffer[track_id].append(embedding)
        
        # Update decision history
        self.decision_history[track_id].append('matched')

    def _update_database_appearance(self, tracking_id: str, bbox: np.ndarray, 
                                   confidence: float, frame_number: int):
        """Update database with new appearance (database operations disabled)."""
        # Database operations disabled - no longer saving to SQLite
        pass
    
    def _save_face_sample(self, frame: np.ndarray, bbox: np.ndarray, track_id: str) -> str:
        """Save face sample image."""
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            face_roi = frame[y1:y2, x1:x2]
            sample_path = os.path.join(self.output_base_dir, "face_samples", f"sample_{track_id}.jpg")
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            cv2.imwrite(sample_path, face_roi)
            return sample_path
        return None
    
    def _cleanup_old_tracks(self, frame_number: int):
        """Remove tracks that haven't been seen for too long."""
        to_remove = []
        
        for track_id, track_info in self.active_tracks.items():
            frames_since_seen = frame_number - track_info['last_seen']
            if frames_since_seen > self.max_disappeared:
                to_remove.append(track_id)
                self.decision_history[track_id].append('lost')
        
        for track_id in to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
            if track_id in self.kalman_trackers:
                del self.kalman_trackers[track_id]
            if track_id in self.face_embeddings_buffer:
                del self.face_embeddings_buffer[track_id]
            if track_id in self.candidate_faces:
                del self.candidate_faces[track_id]
        
        if to_remove:
            print(f"🗑️  Removed {len(to_remove)} old tracks")

    def _confirm_tracks(self, frame_number: int):
        """Confirm candidate tracks that have been consistently detected."""
        confirmed_tracks = []
        
        for track_id in list(self.candidate_faces.keys()):
            if (track_id in self.active_tracks and 
                self.active_tracks[track_id]['appearance_count'] >= self.confirmation_frames):
                
                # Confirm the track
                self.active_tracks[track_id]['confirmed'] = True
                
                # Save to database
                self._save_confirmed_face(track_id, self.candidate_faces[track_id])
                confirmed_tracks.append(track_id)
                
                # Remove from candidates
                del self.candidate_faces[track_id]
                
                print(f"✅ Confirmed new face: {track_id}")
        
        return confirmed_tracks

    def process_faces(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, 
                     frame_number: int, embeddings_data: Dict = None,
                     frame_timestamp_ms: Optional[int] = None) -> Dict[str, Dict]:
        """Process detected faces with enhanced tracking (ORIGINAL LOGIC PRESERVED)."""
        self.frame_count = frame_number
        
        # Extract embeddings for all detections
        detections = []
        for bbox, confidence in zip(boxes, scores):
            embedding = self._extract_enhanced_face_embedding(frame, bbox)
            detections.append((bbox, embedding, confidence))
        
        # Find optimal matches using Hungarian algorithm
        matches = self._find_best_matches(detections, frame_number)
        
        matched_detection_indices = set()
        matched_track_ids = set()
        detection_to_track_mapping = {}  # Map detection index to track ID
        
        # Process matches
        for det_idx, track_id, similarity in matches:
            if det_idx < len(detections):
                bbox, embedding, confidence = detections[det_idx]
                self._update_track(track_id, bbox, embedding, confidence, frame_number)
                matched_detection_indices.add(det_idx)
                matched_track_ids.add(track_id)
                detection_to_track_mapping[det_idx] = track_id
        
        # Create new tracks for unmatched detections
        for det_idx, (bbox, embedding, confidence) in enumerate(detections):
            if det_idx not in matched_detection_indices:
                track_id = self._create_new_track(bbox, embedding, confidence, frame_number, frame)
                matched_track_ids.add(track_id)
                detection_to_track_mapping[det_idx] = track_id
        
        # Update decision history for unmatched tracks
        for track_id in self.active_tracks.keys():
            if track_id not in matched_track_ids:
                self.decision_history[track_id].append('missed')
        
        # Confirm candidate tracks
        self._confirm_tracks(frame_number)
        
        # Cleanup old tracks
        self._cleanup_old_tracks(frame_number)
        
        # Log detections for JSON export
        for track_id in matched_track_ids:
            if track_id in self.active_tracks:
                track_info = self.active_tracks[track_id]
                bbox = track_info['bbox']
                
                # Calculate center, area, and other required fields
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                area_px = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                
                detection_data = {
                    "frame_id": frame_number,
                    "ts_ms": frame_timestamp_ms,
                    "track_id": track_id,
                    "cls": None,  # Not applicable for faces
                    "name": "face",
                    "conf": float(track_info['confidence']),
                    "bbox_xyxy": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "polygon": None,  # Not available for bounding boxes
                    "area_px": area_px,
                    "center": [float(center_x), float(center_y)],
                    "velocity": [0.0, 0.0],  # Would need velocity calculation
                    "direction": "stationary",  # Would need movement analysis
                    "sources": ["retinaface"]
                }
                
                # Add embedding data if available (map from detection index to track ID)
                if embeddings_data and detection_to_track_mapping:
                    # Find the detection index for this track ID
                    det_idx = None
                    for idx, tid in detection_to_track_mapping.items():
                        if tid == track_id:
                            det_idx = idx
                            break
                    
                    # If we found the detection index and have embedding data for it
                    if det_idx is not None and det_idx in embeddings_data:
                        embedding_info = embeddings_data[det_idx]
                        detection_data.update({
                            "embedding": embedding_info.get('embedding'),
                            "embedding_size": embedding_info.get('embedding_size'),
                            "embedding_model": embedding_info.get('model_name'),
                            "detector_backend": embedding_info.get('detector_backend'),
                            "face_confidence": embedding_info.get('face_confidence'),
                            "embedding_timestamp": embedding_info.get('timestamp'),
                            "face_size": embedding_info.get('face_size')
                        })
                self.face_detections_log.append(detection_data)
        
        # Prepare return data
        tracked_faces = {}
        for track_id in matched_track_ids:
            if track_id in self.active_tracks:
                track_info = self.active_tracks[track_id]
                tracked_faces[track_id] = {
                    'bbox': track_info['bbox'],
                    'confidence': track_info['confidence'],
                    'is_new': not track_info.get('confirmed', False),
                    'appearance_count': track_info['appearance_count']
                }
        
        return tracked_faces
    
    def crop_and_save_face(self, frame: np.ndarray, bbox: np.ndarray, tracking_id: str, 
                          frame_number: int, output_dir: str = None):
        """Crop and save face image for confirmed tracks only (if enabled)."""
        # Check if cropped face saving is enabled
        if not self.save_cropped_faces:
            return None
            
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(self.output_base_dir, "faces")
        
        # Only save faces for confirmed tracks
        if (tracking_id in self.active_tracks and 
            self.active_tracks[tracking_id].get('confirmed', False)):
            
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2]
                
                # Create person-specific directory
                person_dir = os.path.join(output_dir, f"person_{tracking_id}")
                os.makedirs(person_dir, exist_ok=True)
                
                face_filename = f"face_frame_{frame_number:06d}.jpg"
                face_path = os.path.join(person_dir, face_filename)
                cv2.imwrite(face_path, face_roi)
                return face_path
        
        return None

    def get_statistics(self) -> Dict:
        """Get enhanced tracking statistics (without database)."""
        total_unique_faces = len(self.face_embeddings)
        total_appearances = sum(track['appearance_count'] for track in self.active_tracks.values())
        
        confirmed_tracks = sum(1 for track in self.active_tracks.values() 
                             if track.get('confirmed', False))
        candidate_tracks = len(self.candidate_faces)
        
        return {
            'total_unique_faces': total_unique_faces,
            'total_appearances': total_appearances,
            'active_tracks': len(self.active_tracks),
            'confirmed_tracks': confirmed_tracks,
            'candidate_tracks': candidate_tracks,
            'similarity_threshold': self.similarity_threshold,
            'spatial_threshold': self.spatial_threshold
        }

    def get_face_summary(self) -> Dict:
        """Get summary of confirmed faces only."""
        face_counts = {}
        
        for track_id, track_info in self.active_tracks.items():
            if track_info.get('confirmed', False):
                face_counts[track_id] = track_info['appearance_count']
        
        return face_counts

    def export_face_data(self, output_path: str = None):
        """Export face detections in the specified JSON format."""
        if output_path is None:
            output_path = os.path.join(self.output_base_dir, "face_data_export.json")
        
        # Export all logged detections in the new format
        with open(output_path, 'w') as f:
            json.dump(self.face_detections_log, f, indent=2)
        
        print(f"Face detection data exported to {output_path} with {len(self.face_detections_log)} detections")
