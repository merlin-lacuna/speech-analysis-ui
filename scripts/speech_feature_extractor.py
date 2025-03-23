import numpy as np
import librosa
import faiss
import json
import os
import torch
import time
import logging
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   stream=sys.stdout)  # Ensure logs go to stdout
logger = logging.getLogger(__name__)

class LogCapture:
    def __init__(self):
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logs = ""
        
    def __enter__(self):
        logger.addHandler(self.handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.removeHandler(self.handler)
        self.logs = self.log_stream.getvalue()
        self.log_stream.close()
        
    def get_logs(self):
        return self.logs

class HybridFeatureExtractor:
    _instance: Optional['HybridFeatureExtractor'] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HybridFeatureExtractor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Skip initialization if already done
        if HybridFeatureExtractor._initialized:
            return
            
        logger.info("Initializing CPU-only feature extractor...")
        self.feature_dim = 40  # 20 means + 20 stds
        self.use_gpu = False
        self.index = faiss.IndexFlatL2(self.feature_dim)
        
        # Store labels and mappings
        self.labels: List[str] = []
        self.label_to_idx: Dict[str, int] = {}
        
        # Initialize PyTorch device
        logger.info("Initializing PyTorch device...")
        self.device = torch.device('cpu')  # Force CPU
        logger.info(f"Using PyTorch device: {self.device}")
        
        HybridFeatureExtractor._initialized = True
        logger.info("Initialization complete")

    def extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract audio features using GPU acceleration"""
        # Convert to PyTorch tensor and move to GPU
        y_tensor = torch.from_numpy(y).to(self.device)
        
        # Extract MFCCs (on CPU as librosa doesn't support GPU)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_tensor = torch.from_numpy(mfccs).to(self.device)
        
        # Compute statistics on GPU
        mfcc_means = mfccs_tensor.mean(dim=1)
        mfcc_stds = mfccs_tensor.std(dim=1)
        
        # Combine features
        features = torch.cat([mfcc_means, mfcc_stds])
        
        # Move back to CPU and convert to numpy
        return features.cpu().numpy().astype(np.float32)

    def extract_all_features(self, audio_path: str) -> Tuple[Optional[np.ndarray], str]:
        """Extract all features from an audio file"""
        log_capture = LogCapture()
        with log_capture:
            try:
                # Load audio with optimized settings
                logger.info(f"Loading audio file: {audio_path}")
                y, sr = librosa.load(audio_path, sr=None, duration=5.0)  # Limit to 5 seconds for faster processing
                
                # Extract features
                features = self.extract_features(y, sr)
                logger.info(f"Feature shape: {features.shape}")
                return features, log_capture.get_logs()
                
            except Exception as e:
                logger.error(f"Error in feature extraction: {str(e)}")
                return None, log_capture.get_logs()

    def add_sample(self, audio_path: str, label: str):
        """Add a labeled sample to the database"""
        features, _ = self.extract_all_features(audio_path)
        
        # Add to index
        self.index.add(features.reshape(1, -1))
        
        # Store label
        self.labels.append(label)
        if label not in self.label_to_idx:
            self.label_to_idx[label] = len(self.label_to_idx)

    def find_closest_match(self, audio_path: str) -> str:
        """Find the closest matching sample and return logs"""
        log_capture = LogCapture()
        with log_capture:
            try:
                start_time = time.time()
                logger.info(f"Processing audio file: {audio_path}")
                
                # Extract features from input
                features, extract_logs = self.extract_all_features(audio_path)
                if features is None:
                    return json.dumps({
                        "result": None,
                        "logs": log_capture.get_logs() + "\n" + extract_logs
                    })
                    
                logger.info(f"Features extracted in {time.time() - start_time:.2f} seconds")
                
                # Search in index
                search_start = time.time()
                D, I = self.index.search(features.reshape(1, -1), k=2)
                logger.info(f"Search completed in {time.time() - search_start:.2f} seconds")
                
                # Get label and distance
                closest_label = self.labels[I[0][0]]
                distance = D[0][0]
                
                # Calculate confidence score
                if len(D[0]) > 1:
                    second_best_distance = D[0][1]
                    confidence = 1.0 - (distance / second_best_distance)
                    confidence = max(0.0, min(1.0, confidence * 1.5))
                else:
                    confidence = max(0.0, min(1.0, 1.0 - distance / 1000.0))
                
                logger.info(f"Found match: {closest_label} with confidence {confidence:.2f}")
                
                # Return JSON with both result and logs
                return json.dumps({
                    "result": {
                        "label": closest_label,
                        "confidence": float(confidence)
                    },
                    "logs": log_capture.get_logs() + "\n" + extract_logs
                })
                
            except Exception as e:
                logger.error(f"Error in find_closest_match: {str(e)}")
                return json.dumps({
                    "result": None,
                    "logs": log_capture.get_logs()
                })

    def save_database(self, path: str):
        """Save the feature database and labels"""
        if self.use_gpu:
            # Convert GPU index to CPU for saving
            print("Converting GPU index to CPU for saving...")
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, f"{path}_index.faiss")
        else:
            faiss.write_index(self.index, f"{path}_index.faiss")
        
        # Save labels and mappings
        with open(f"{path}_labels.json", 'w') as f:
            json.dump({
                'labels': self.labels,
                'label_to_idx': self.label_to_idx
            }, f)

    def load_database(self, path: str) -> str:
        """Load a saved feature database and labels, return logs"""
        log_capture = LogCapture()
        with log_capture:
            logger.info(f"Loading database from {path}...")
            
            # Check if database files exist
            index_path = f"{path}_index.faiss"
            labels_path = f"{path}_labels.json"
            
            if not os.path.exists(index_path):
                logger.error(f"Index file not found at {index_path}")
                return json.dumps({
                    "result": False,
                    "logs": log_capture.get_logs()
                })
                
            if not os.path.exists(labels_path):
                logger.error(f"Labels file not found at {labels_path}")
                return json.dumps({
                    "result": False,
                    "logs": log_capture.get_logs()
                })
            
            try:
                # Load CPU index with timeout
                start_time = time.time()
                logger.info("Loading FAISS index...")
                cpu_index = faiss.read_index(index_path)
                
                if self.use_gpu:
                    try:
                        logger.info("Converting index to GPU...")
                        if time.time() - start_time > 30:  # 30 second timeout
                            raise TimeoutError("Index loading took too long")
                            
                        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
                        logger.info("Successfully converted index to GPU")
                    except Exception as e:
                        logger.error(f"Error converting to GPU index: {str(e)}")
                        logger.warning("Using CPU index instead")
                        self.use_gpu = False
                        self.index = cpu_index
                else:
                    self.index = cpu_index
                
                # Load labels and mappings
                logger.info("Loading labels...")
                with open(labels_path, 'r') as f:
                    data = json.load(f)
                    self.labels = data['labels']
                    self.label_to_idx = data['label_to_idx']
                logger.info(f"Database loaded successfully with {len(self.labels)} samples")
                
                return json.dumps({
                    "result": True,
                    "logs": log_capture.get_logs()
                })
                
            except Exception as e:
                logger.error(f"Error loading database: {str(e)}")
                return json.dumps({
                    "result": False,
                    "logs": log_capture.get_logs()
                })

# Create a function to get or create the extractor
def get_feature_extractor() -> HybridFeatureExtractor:
    try:
        extractor = HybridFeatureExtractor()
        return extractor
    except Exception as e:
        logger.error(f"Failed to create feature extractor: {str(e)}")
        raise

if __name__ == "__main__":
    # If running directly, initialize and handle command line arguments
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python speech_feature_extractor.py <command> [args...]")
        sys.exit(1)
        
    extractor = get_feature_extractor()
    command = sys.argv[1]
    
    if command == "load":
        if len(sys.argv) < 3:
            print("Usage: python speech_feature_extractor.py load <database_path>")
            sys.exit(1)
        print(extractor.load_database(sys.argv[2]))
        
    elif command == "match":
        if len(sys.argv) < 3:
            print("Usage: python speech_feature_extractor.py match <audio_path>")
            sys.exit(1)
        print(extractor.find_closest_match(sys.argv[2]))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1) 