"""
Face Embedding Generator using ArcFace from DeepFace
===================================================
This module provides face embedding generation capabilities using ArcFace model
from the DeepFace library for high-quality face recognition embeddings.
"""

import os
import numpy as np
import cv2
import json
from datetime import datetime
import logging

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available. Please install: pip install deepface")

class ArcFaceEmbeddingGenerator:
    """
    ArcFace embedding generator using DeepFace library.
    
    Supports multiple ArcFace model variants:
    - ArcFace for high accuracy
    - Facenet512 for balance of speed and accuracy
    - OpenFace for lightweight applications
    """
    
    def __init__(self, model_name="ArcFace", detector_backend="opencv", 
                 min_face_size=32, enable_alignment=True):
        """
        Initialize ArcFace embedding generator.
        
        Args:
            model_name (str): Model to use - 'ArcFace', 'Facenet512', 'OpenFace'
            detector_backend (str): Detection backend - 'opencv', 'retinaface', 'mtcnn'
            min_face_size (int): Minimum face size in pixels
            enable_alignment (bool): Enable face alignment
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace not available. Please install: pip install deepface")
        
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.min_face_size = min_face_size
        self.enable_alignment = enable_alignment
        
        # Model specifications
        self.model_specs = {
            "ArcFace": {"embedding_size": 512, "description": "High accuracy ArcFace model"},
            "Facenet512": {"embedding_size": 512, "description": "Balanced FaceNet model"},
            "OpenFace": {"embedding_size": 128, "description": "Lightweight OpenFace model"},
            "Facenet": {"embedding_size": 128, "description": "Original FaceNet model"}
        }
        
        print(f"🎯 Initialized ArcFace Embedding Generator")
        print(f"   Model: {self.model_name}")
        print(f"   Detector: {self.detector_backend}")
        print(f"   Embedding size: {self.model_specs.get(model_name, {}).get('embedding_size', 'Unknown')}")
    
    def generate_embedding(self, face_image):
        """
        Generate face embedding using ArcFace model.
        
        Args:
            face_image (numpy.ndarray): Face image (BGR format from OpenCV)
            
        Returns:
            dict: Dictionary containing embedding and metadata, or None if failed
        """
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            # Validate input image
            if face_image is None or face_image.size == 0:
                return None
            
            # Check minimum face size
            if face_image.shape[0] < self.min_face_size or face_image.shape[1] < self.min_face_size:
                return None
            
            # Convert BGR to RGB for DeepFace (if needed)
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Generate embedding using DeepFace
            embeddings = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Don't fail if face detection fails
                align=self.enable_alignment,
                normalization="base"  # Use base normalization
            )
            
            if embeddings and len(embeddings) > 0:
                embedding_vector = embeddings[0]['embedding']
                
                # Create embedding result
                result = {
                    'embedding': embedding_vector,
                    'embedding_size': len(embedding_vector),
                    'model_name': self.model_name,
                    'detector_backend': self.detector_backend,
                    'face_confidence': embeddings[0].get('face_confidence', None),
                    'timestamp': datetime.now().isoformat(),
                    'face_size': {'width': face_image.shape[1], 'height': face_image.shape[0]}
                }
                
                return result
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Error generating ArcFace embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, face_images):
        """
        Generate embeddings for multiple face images.
        
        Args:
            face_images (list): List of face images (numpy arrays)
            
        Returns:
            list: List of embedding results
        """
        embeddings = []
        
        for i, face_image in enumerate(face_images):
            print(f"🔄 Processing face {i+1}/{len(face_images)}")
            embedding = self.generate_embedding(face_image)
            embeddings.append(embedding)
        
        return embeddings
    
    def save_embeddings_to_json(self, embeddings, output_path):
        """
        Save embeddings to JSON file.
        
        Args:
            embeddings (list): List of embedding results
            output_path (str): Output JSON file path
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_embeddings = []
            for embedding in embeddings:
                if embedding is not None:
                    serializable_embedding = embedding.copy()
                    if 'embedding' in serializable_embedding:
                        serializable_embedding['embedding'] = serializable_embedding['embedding'].tolist()
                    serializable_embeddings.append(serializable_embedding)
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump({
                    'embeddings': serializable_embeddings,
                    'metadata': {
                        'total_embeddings': len(serializable_embeddings),
                        'model_name': self.model_name,
                        'detector_backend': self.detector_backend,
                        'generated_at': datetime.now().isoformat()
                    }
                }, f, indent=2)
            
            print(f"✅ Embeddings saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
    
    def get_model_info(self):
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'detector_backend': self.detector_backend,
            'embedding_size': self.model_specs.get(self.model_name, {}).get('embedding_size'),
            'description': self.model_specs.get(self.model_name, {}).get('description'),
            'min_face_size': self.min_face_size,
            'alignment_enabled': self.enable_alignment
        }

def create_embedding_generator(config=None):
    """
    Factory function to create embedding generator from configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        ArcFaceEmbeddingGenerator: Configured embedding generator
    """
    if config is None:
        config = {}
    
    model_name = config.get('embedding_model', 'ArcFace')
    detector_backend = config.get('embedding_detector_backend', 'opencv')
    min_face_size = config.get('min_face_size', 32)
    enable_alignment = config.get('enable_face_alignment', True)
    
    return ArcFaceEmbeddingGenerator(
        model_name=model_name,
        detector_backend=detector_backend,
        min_face_size=min_face_size,
        enable_alignment=enable_alignment
    )

# Example usage
if __name__ == "__main__":
    # Test the embedding generator
    generator = ArcFaceEmbeddingGenerator()
    print("🧪 ArcFace Embedding Generator Test")
    print(f"📊 Model Info: {generator.get_model_info()}")
