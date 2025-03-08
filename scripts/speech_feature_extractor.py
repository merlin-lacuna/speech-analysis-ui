import numpy as np
import librosa
import faiss
import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

class HybridFeatureExtractor:
    def __init__(self):
        # Initialize FAISS index with a smaller feature dimension
        self.feature_dim = 40  # Reduced feature dimension
        self.index = faiss.IndexFlatL2(self.feature_dim)
        
        # Store labels and their mappings
        self.labels: List[str] = []
        self.label_to_idx: Dict[str, int] = {}

    def extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract basic audio features"""
        # Basic features that we know will work
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Combine features
        features = np.concatenate([mfcc_means, mfcc_stds])
        return features.astype(np.float32)

    def extract_all_features(self, audio_path: str) -> np.ndarray:
        """Extract all features from an audio file"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        try:
            # Extract features
            features = self.extract_features(y, sr)
            
            # Debug print
            print(f"Feature shape: {features.shape}")
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

    def add_sample(self, audio_path: str, label: str):
        """Add a labeled sample to the database"""
        features = self.extract_all_features(audio_path)
        
        # Add to FAISS index
        self.index.add(features.reshape(1, -1))
        
        # Store label
        self.labels.append(label)
        if label not in self.label_to_idx:
            self.label_to_idx[label] = len(self.label_to_idx)

    def find_closest_match(self, audio_path: str) -> Tuple[str, float]:
        """Find the closest matching sample for a given audio input"""
        # Extract features from input
        features = self.extract_all_features(audio_path)
        
        # Search in FAISS index
        D, I = self.index.search(features.reshape(1, -1), k=2)  # Get top 2 matches
        
        # Get label and distance
        closest_label = self.labels[I[0][0]]
        distance = D[0][0]
        
        # Calculate confidence score
        if len(D[0]) > 1:
            # Compare with second best match to get relative confidence
            second_best_distance = D[0][1]
            # If distance to best match is much smaller than to second best, confidence is high
            confidence = 1.0 - (distance / second_best_distance)
            # Clip to range [0, 1] and scale to make more meaningful
            confidence = max(0.0, min(1.0, confidence * 1.5))
        else:
            # Fallback if only one match exists
            confidence = max(0.0, min(1.0, 1.0 - distance / 1000.0))
        
        return closest_label, confidence

    def save_database(self, path: str):
        """Save the feature database and labels"""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}_index.faiss")
        
        # Save labels and mappings
        with open(f"{path}_labels.json", 'w') as f:
            json.dump({
                'labels': self.labels,
                'label_to_idx': self.label_to_idx
            }, f)

    def load_database(self, path: str):
        """Load a saved feature database and labels"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}_index.faiss")
        
        # Load labels and mappings
        with open(f"{path}_labels.json", 'r') as f:
            data = json.load(f)
            self.labels = data['labels']
            self.label_to_idx = data['label_to_idx'] 