import numpy as np
import librosa
import faiss
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import torch
import torchaudio
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert
import logging

# Import SpeechBrain components
import speechbrain as sb
from speechbrain.inference import VAD
from speechbrain.inference import EncoderClassifier
from speechbrain.lobes.features import MFCC, Fbank
# Note: We're already using our own compute_deltas function defined below

def compute_deltas(features: np.ndarray, width: int = 9) -> np.ndarray:
    """Compute delta features from a feature matrix"""
    padded = np.pad(features, ((0, 0), (width // 2, width // 2)), mode='edge')
    windows = np.array([padded[:, i:i + features.shape[1]] for i in range(width)])
    weights = np.arange(-(width // 2), (width // 2) + 1)
    deltas = np.tensordot(windows, weights, axes=(0, 0)) / weights.dot(weights)
    return deltas

@dataclass
class SpeechFeatures:
    rhythm_features: np.ndarray
    energy_features: np.ndarray
    pitch_features: np.ndarray
    pause_features: np.ndarray
    voice_quality_features: np.ndarray
    speech_rate_features: np.ndarray

class FeatureExtractorBase:
    """Base class for feature extractors"""
    def __init__(self):
        # Initialize separate FAISS indices for different feature categories
        self.feature_dims = {
            'rhythm': 32,
            'energy': 24,
            'pitch': 28,
            'pause': 16,
            'voice_quality': 40,
            'speech_rate': 20
        }
        
        self.indices = {
            category: faiss.IndexFlatL2(dim) 
            for category, dim in self.feature_dims.items()
        }
        
        # Store labels and their mappings
        self.labels: List[str] = []
        self.label_to_idx: Dict[str, int] = {}
        
        # Initialize scalers for each feature category
        self.scalers = {
            category: StandardScaler() 
            for category in self.feature_dims.keys()
        }
    
    def extract_all_features(self, audio_path: str) -> SpeechFeatures:
        """Extract all feature categories from an audio file - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def add_sample(self, audio_path: str, label: str):
        """Add a labeled sample to the database"""
        try:
            # Extract features
            features = self.extract_all_features(audio_path)
            
            # Track successful additions
            successful_additions = []
            
            # Add features to respective indices
            for category, index in self.indices.items():
                try:
                    category_features = getattr(features, f"{category}_features")
                    
                    # Ensure features are properly shaped
                    features_reshaped = category_features.reshape(1, -1).astype(np.float32)
                    
                    # Handle NaN values
                    if np.any(np.isnan(features_reshaped)):
                        logging.warning(f"NaN values detected in {category} features")
                        # Instead of replacing with zeros, skip this category
                        continue
                    
                    # Normalize features using the scaler
                    if len(self.labels) > 0:  # If we have existing samples
                        # Partial fit to update scaler with new data
                        self.scalers[category].partial_fit(features_reshaped)
                    else:  # First sample
                        self.scalers[category].fit(features_reshaped)
                    
                    # Transform features using the fitted scaler
                    features_normalized = self.scalers[category].transform(features_reshaped)
                    
                    # Verify the normalized features
                    if np.any(np.isnan(features_normalized)) or np.any(np.isinf(features_normalized)):
                        logging.warning(f"Invalid values after normalization in {category} features")
                        continue
                    
                    # Add to index
                    index.add(features_normalized)
                    successful_additions.append(category)
                    
                except Exception as e:
                    logging.error(f"Error adding {category} features: {str(e)}")
                    continue
            
            # Only proceed if we successfully added at least some features
            if not successful_additions:
                raise ValueError("Failed to add features for any category")
            
            # Store label
            self.labels.append(label)
            if label not in self.label_to_idx:
                self.label_to_idx[label] = len(self.label_to_idx)
            
            # Save after successful ingestion
            try:
                self.save_database('speech_database')
                logging.info(f"Successfully added and saved sample with label: {label}")
                logging.info(f"Successfully processed categories: {', '.join(successful_additions)}")
            except Exception as e:
                logging.error(f"Error saving database: {str(e)}")
                # If we can't save, we should remove the sample we just added
                self._remove_last_sample()
                raise
                
        except Exception as e:
            logging.error(f"Error in sample addition: {str(e)}")
            raise

    def _remove_last_sample(self):
        """Remove the last added sample from all indices"""
        if not self.labels:
            return
            
        # Remove from each index
        for category, index in self.indices.items():
            try:
                # FAISS doesn't support direct removal, so we need to rebuild the index
                if index.ntotal > 1:
                    # Get all vectors except the last one
                    all_vectors = index.reconstruct_n(0, index.ntotal - 1)
                    
                    # Create new index
                    new_index = faiss.IndexFlatL2(self.feature_dims[category])
                    new_index.add(all_vectors)
                    
                    # Replace old index
                    self.indices[category] = new_index
            except Exception as e:
                logging.error(f"Error removing last sample from {category} index: {str(e)}")
        
        # Remove last label
        self.labels.pop()

    def find_closest_match(self, audio_path: str) -> Tuple[str, Dict[str, float]]:
        """Find the closest matching sample with detailed category scores"""
        # Extract features from input
        features = self.extract_all_features(audio_path)
        
        # Check if we have any samples in the database
        if not self.labels:
            logging.warning("No samples in database yet. Please add some samples first.")
            return "no_samples", {category: 0.0 for category in self.feature_dims.keys()}
        
        # Calculate similarity scores for each category
        category_scores = {}
        overall_confidence = 0.0
        valid_categories = 0
        
        for category, index in self.indices.items():
            try:
                category_features = getattr(features, f"{category}_features")
                features_reshaped = category_features.reshape(1, -1).astype(np.float32)
                
                # Skip if features contain NaN
                if np.any(np.isnan(features_reshaped)):
                    logging.warning(f"NaN values in {category} features, skipping")
                    category_scores[category] = 0.0
                    continue
                
                # Normalize using the same scaler used during training
                features_normalized = self.scalers[category].transform(features_reshaped)
                
                # Skip if normalization produced invalid values
                if np.any(np.isnan(features_normalized)) or np.any(np.isinf(features_normalized)):
                    logging.warning(f"Invalid values after normalization in {category} features")
                    category_scores[category] = 0.0
                    continue
                
                # Search in the index
                D, I = index.search(features_normalized, k=min(2, len(self.labels)))
                
                # Calculate confidence score for this category
                if len(D[0]) > 1:
                    best_distance = D[0][0]
                    second_best_distance = D[0][1]
                    confidence = 1.0 - (best_distance / (second_best_distance + 1e-6))
                    confidence = max(0.0, min(1.0, confidence * 1.5))
                else:
                    confidence = max(0.0, min(1.0, 1.0 - D[0][0] / 1000.0))
                
                category_scores[category] = confidence
                overall_confidence += confidence
                valid_categories += 1
                
            except Exception as e:
                logging.error(f"Error calculating {category} score: {str(e)}")
                category_scores[category] = 0.0
        
        # Average the confidence scores only for valid categories
        if valid_categories > 0:
            overall_confidence /= valid_categories
        else:
            logging.warning("No valid categories found for matching")
            return "unknown", category_scores
        
        # Get the most common label from valid categories
        label_votes = {}
        for category, index in self.indices.items():
            try:
                if category_scores[category] > 0.0:  # Only consider categories with valid scores
                    category_features = getattr(features, f"{category}_features")
                    features_normalized = self.scalers[category].transform(
                        category_features.reshape(1, -1).astype(np.float32)
                    )
                    _, I = index.search(features_normalized, k=1)
                    voted_label = self.labels[I[0][0]]
                    label_votes[voted_label] = label_votes.get(voted_label, 0) + 1
            except Exception as e:
                logging.error(f"Error getting label for {category}: {str(e)}")
                continue
        
        if label_votes:
            closest_label = max(label_votes.items(), key=lambda x: x[1])[0]
        else:
            closest_label = "unknown"
        
        return closest_label, category_scores

    def save_database(self, path: str):
        """Save the feature database and labels"""
        # Save FAISS indices
        for category, index in self.indices.items():
            faiss.write_index(index, f"{path}_{category}_index.faiss")
        
        # Save labels and mappings
        with open(f"{path}_labels.json", 'w') as f:
            json.dump({
                'labels': self.labels,
                'label_to_idx': self.label_to_idx
            }, f)
        
        # Save fitted scalers
        scaler_data = {}
        for category, scaler in self.scalers.items():
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                scaler_data[category] = {
                    'mean': scaler.mean_.tolist() if scaler.mean_ is not None else None,
                    'scale': scaler.scale_.tolist() if scaler.scale_ is not None else None,
                    'var': scaler.var_.tolist() if hasattr(scaler, 'var_') and scaler.var_ is not None else None,
                    'n_samples_seen': int(scaler.n_samples_seen_) if hasattr(scaler, 'n_samples_seen_') else 0
                }
        
        if scaler_data:
            with open(f"{path}_scalers.json", 'w') as f:
                json.dump(scaler_data, f)

    def load_database(self, path: str):
        """Load a saved feature database and labels"""
        # Load FAISS indices
        for category in self.feature_dims.keys():
            self.indices[category] = faiss.read_index(f"{path}_{category}_index.faiss")
        
        # Load labels and mappings
        with open(f"{path}_labels.json", 'r') as f:
            data = json.load(f)
            self.labels = data['labels']
            self.label_to_idx = data['label_to_idx']
        
        # Load fitted scalers if they exist
        scaler_path = f"{path}_scalers.json"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
                
            for category, data in scaler_data.items():
                if category in self.scalers:
                    scaler = self.scalers[category]
                    if data['mean'] is not None:
                        scaler.mean_ = np.array(data['mean'])
                    if data['scale'] is not None:
                        scaler.scale_ = np.array(data['scale'])
                    if data['var'] is not None:
                        scaler.var_ = np.array(data['var'])
                    if data['n_samples_seen'] > 0:
                        scaler.n_samples_seen_ = data['n_samples_seen']
        else:
            # Initialize scalers with identity transformation if no saved data
            logging.warning("No saved scaler data found, initializing with identity transformation")
            for category in self.feature_dims.keys():
                scaler = self.scalers[category]
                dim = self.feature_dims[category]
                scaler.mean_ = np.zeros(dim)
                scaler.scale_ = np.ones(dim)
                scaler.var_ = np.ones(dim)
                scaler.n_samples_seen_ = 1

class LibrosaFeatureExtractor(FeatureExtractorBase):
    """Traditional signal processing approach using librosa"""
    def __init__(self):
        super().__init__()
        
    def extract_all_features(self, audio_path: str) -> SpeechFeatures:
        """Extract all feature categories from an audio file"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        try:
            # Extract all feature categories
            features = SpeechFeatures(
                rhythm_features=self.extract_rhythm_features(y, sr),
                energy_features=self.extract_energy_features(y, sr),
                pitch_features=self.extract_pitch_features(y, sr),
                pause_features=self.extract_pause_features(y, sr),
                voice_quality_features=self.extract_voice_quality_features(y, sr),
                speech_rate_features=self.extract_speech_rate_features(y, sr)
            )
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

    def extract_rhythm_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract rhythm-related features"""
        try:
            # Zero-crossing rate for rhythm analysis
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Rhythm regularity features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_delta = compute_deltas(mfcc)  # Use our custom delta function
            
            # Process MFCC deltas to ensure 1D
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            mfcc_delta_std = np.std(mfcc_delta, axis=1)
            
            # Debug print
            print("\nRhythm features shapes:")
            print(f"zcr: {zcr.shape}")
            print(f"tempo: {np.array([tempo]).shape}")  # Debug tempo shape
            print(f"mfcc_delta_mean: {mfcc_delta_mean.shape}")
            print(f"mfcc_delta_std: {mfcc_delta_std.shape}")
            
            # Calculate zcr statistics first to ensure 1D
            zcr_mean = np.mean(zcr, axis=1)[0]  # Take first channel if stereo
            zcr_std = np.std(zcr, axis=1)[0]
            
            # Create feature arrays one by one for debugging
            feature_arrays = [
                np.array([zcr_mean]),  # Now properly 1D
                np.array([zcr_std]),
                np.array([float(tempo)]),  # Ensure tempo is a float
                mfcc_delta_mean.ravel(),  # Already 1D from mean operation
                mfcc_delta_std.ravel(),   # Already 1D from std operation
                np.array([np.mean(np.abs(compute_deltas(mfcc_delta_mean.reshape(-1, 1)).flatten()))]),  # Overall rhythm change
                np.array([skew(mfcc_delta.ravel())])  # Overall skewness
            ]
            
            # Print shapes for debugging
            for i, arr in enumerate(feature_arrays):
                print(f"Array {i}: shape {arr.shape}, dims {arr.ndim}")
            
            # Concatenate arrays
            features = np.concatenate(feature_arrays)
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['rhythm']:
                features = np.pad(features, (0, self.feature_dims['rhythm'] - len(features)))
            else:
                features = features[:self.feature_dims['rhythm']]
                
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in rhythm features: {str(e)}")
            raise

    def extract_energy_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract energy-related features"""
        try:
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]  # Get the first channel
            
            # Spectral energy
            spec = np.abs(librosa.stft(y))
            spec_energy = np.sum(spec, axis=0)
            
            # Debug print
            print("\nEnergy features shapes:")
            print(f"rms: {rms.shape}")
            print(f"spec_energy: {spec_energy.shape}")
            
            # Energy statistics - ensure all are 1D arrays
            features = np.concatenate([
                np.array([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)]),
                np.array([np.mean(spec_energy), np.std(spec_energy)]),
                np.array([skew(rms), kurtosis(rms)]),
                np.array(np.percentile(rms, [25, 50, 75]))
            ])
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['energy']:
                features = np.pad(features, (0, self.feature_dims['energy'] - len(features)))
            else:
                features = features[:self.feature_dims['energy']]
                
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in energy features: {str(e)}")
            raise

    def extract_pitch_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch-independent features"""
        try:
            # Pitch tracking
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Debug print
            print("\nPitch features shapes:")
            print(f"pitches: {pitches.shape}")
            
            # Get pitch contour (normalized)
            pitch_contour = np.mean(pitches, axis=0)
            if len(pitch_contour) > 0:  # Check if we got any pitch values
                pitch_contour = (pitch_contour - np.mean(pitch_contour)) / (np.std(pitch_contour) + 1e-6)
            else:
                pitch_contour = np.zeros(1)  # Fallback if no pitch detected
            
            print(f"pitch_contour: {pitch_contour.shape}")
            
            # Extract pitch variation features - ensure all are arrays
            features = np.concatenate([
                np.array([np.std(pitch_contour)]),
                np.array([np.mean(np.abs(np.diff(pitch_contour)))]),
                np.array([skew(pitch_contour), kurtosis(pitch_contour)]),
                np.array(np.percentile(pitch_contour, [10, 30, 50, 70, 90]))
            ])
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['pitch']:
                features = np.pad(features, (0, self.feature_dims['pitch'] - len(features)))
            else:
                features = features[:self.feature_dims['pitch']]
                
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in pitch features: {str(e)}")
            raise

    def extract_pause_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract pause pattern features"""
        try:
            # Energy-based silence detection
            rms = librosa.feature.rms(y=y)[0]
            silence_threshold = np.mean(rms) * 0.1
            is_silence = rms < silence_threshold
            
            # Calculate pause statistics
            pause_lengths = []
            current_pause = 0
            
            for is_silent in is_silence:
                if is_silent:
                    current_pause += 1
                elif current_pause > 0:
                    pause_lengths.append(current_pause)
                    current_pause = 0
            
            # Add final pause if we ended in silence
            if current_pause > 0:
                pause_lengths.append(current_pause)
            
            # Convert to numpy array and calculate features
            if pause_lengths:
                pause_lengths = np.array(pause_lengths, dtype=np.float32)
                features = np.array([
                    float(len(pause_lengths)),  # Number of pauses
                    float(np.mean(pause_lengths)),  # Average pause length
                    float(np.std(pause_lengths)) if len(pause_lengths) > 1 else 0.0,  # Pause variation
                    float(np.sum(is_silence)) / float(len(is_silence)),  # Silence ratio
                ], dtype=np.float32)
            else:
                features = np.zeros(4, dtype=np.float32)
            
            # Debug print
            print("\nPause features:")
            print(f"Number of pauses: {features[0]}")
            print(f"Average pause length: {features[1]:.2f}")
            print(f"Pause variation: {features[2]:.2f}")
            print(f"Silence ratio: {features[3]:.2f}")
            
            # Pad to match expected dimension
            if len(features) < self.feature_dims['pause']:
                features = np.pad(features, (0, self.feature_dims['pause'] - len(features)))
            else:
                features = features[:self.feature_dims['pause']]
            
            return features
        except Exception as e:
            print(f"Error in pause features: {str(e)}")
            # Return zero features as fallback
            return np.zeros(self.feature_dims['pause'], dtype=np.float32)

    def extract_voice_quality_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract voice quality features"""
        try:
            # Spectral features - ensure we get 1D arrays
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # Debug print
            print("\nVoice quality features shapes:")
            print(f"spectral_centroid: {spectral_centroid.shape}")
            print(f"spectral_bandwidth: {spectral_bandwidth.shape}")
            print(f"spectral_rolloff: {spectral_rolloff.shape}")
            
            # MFCC-based voice quality
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            print(f"mfcc: {mfcc.shape}")
            
            # Ensure all features are 1D arrays before concatenation
            features = np.concatenate([
                np.array([np.mean(spectral_centroid.ravel())]),
                np.array([np.std(spectral_centroid.ravel())]),
                np.array([np.mean(spectral_bandwidth.ravel())]),
                np.array([np.std(spectral_bandwidth.ravel())]),
                np.array([np.mean(spectral_rolloff.ravel())]),
                np.array([np.std(spectral_rolloff.ravel())]),
                np.mean(mfcc, axis=1).ravel()  # Ensure 1D
            ])
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['voice_quality']:
                features = np.pad(features, (0, self.feature_dims['voice_quality'] - len(features)))
            else:
                features = features[:self.feature_dims['voice_quality']]
                
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in voice quality features: {str(e)}")
            raise

    def extract_speech_rate_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract speech rate features"""
        try:
            # Debug print
            print("\nSpeech rate features shapes:")
            
            # Envelope-based rate analysis using scipy's hilbert
            envelope = np.abs(hilbert(y.ravel()))  # Make sure input is 1D
            print(f"envelope: {envelope.shape}")
            
            # Reshape for RMS calculation and ensure non-empty
            envelope_frames = envelope.reshape(-1, 1)
            envelope_smooth = librosa.feature.rms(y=envelope_frames, frame_length=2048)[0]
            print(f"envelope_smooth before reshape: {envelope_smooth.shape}")
            
            # Handle empty or invalid values
            if len(envelope_smooth) == 0 or np.any(np.isnan(envelope_smooth)):
                print("Warning: Empty or invalid envelope, using zeros")
                # Return zero features with correct dimension
                features = np.zeros(self.feature_dims['speech_rate'], dtype=np.float32)
                return features
            
            # Ensure envelope_smooth is 1D
            envelope_smooth = envelope_smooth.ravel()
            print(f"envelope_smooth after reshape: {envelope_smooth.shape}")
            
            # Calculate statistics - ensure all are scalar values with fallbacks
            try:
                env_mean = float(np.mean(envelope_smooth)) if len(envelope_smooth) > 0 else 0.0
            except:
                env_mean = 0.0
                
            try:
                env_std = float(np.std(envelope_smooth)) if len(envelope_smooth) > 0 else 0.0
            except:
                env_std = 0.0
                
            try:
                env_skew = float(skew(envelope_smooth)) if len(envelope_smooth) > 0 else 0.0
            except:
                env_skew = 0.0
                
            try:
                env_rate = float(np.mean(np.abs(np.diff(envelope_smooth)))) if len(envelope_smooth) > 1 else 0.0
            except:
                env_rate = 0.0
            
            print(f"Statistics: mean={env_mean}, std={env_std}, skew={env_skew}, rate={env_rate}")
            
            # Create feature arrays one by one for debugging
            feature_arrays = [
                np.array([env_mean], dtype=np.float32),
                np.array([env_std], dtype=np.float32),
                np.array([env_skew], dtype=np.float32),
                np.array([env_rate], dtype=np.float32)
            ]
            
            # Print shapes for debugging
            print("\nSpeech rate feature array shapes:")
            for i, arr in enumerate(feature_arrays):
                print(f"Array {i}: shape {arr.shape}, dims {arr.ndim}")
            
            # Concatenate arrays
            features = np.concatenate(feature_arrays)
            print(f"Concatenated features: {features.shape}")
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['speech_rate']:
                features = np.pad(features, (0, self.feature_dims['speech_rate'] - len(features)))
            else:
                features = features[:self.feature_dims['speech_rate']]
            
            print(f"Final features: {features.shape}")
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in speech rate features: {str(e)}")
            raise

    def extract_all_features(self, audio_path: str) -> SpeechFeatures:
        """Extract all feature categories from an audio file"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        try:
            # Extract all feature categories
            features = SpeechFeatures(
                rhythm_features=self.extract_rhythm_features(y, sr),
                energy_features=self.extract_energy_features(y, sr),
                pitch_features=self.extract_pitch_features(y, sr),
                pause_features=self.extract_pause_features(y, sr),
                voice_quality_features=self.extract_voice_quality_features(y, sr),
                speech_rate_features=self.extract_speech_rate_features(y, sr)
            )
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

    def add_sample(self, audio_path: str, label: str):
        """Add a labeled sample to the database"""
        features = self.extract_all_features(audio_path)
        
        # Add features to respective indices
        for category, index in self.indices.items():
            try:
                category_features = getattr(features, f"{category}_features")
                # Ensure features are properly shaped for FAISS
                features_reshaped = category_features.reshape(1, -1).astype(np.float32)
                if np.any(np.isnan(features_reshaped)):
                    print(f"Warning: NaN values in {category} features, replacing with zeros")
                    features_reshaped = np.nan_to_num(features_reshaped, 0.0)
                index.add(features_reshaped)
            except Exception as e:
                print(f"Warning: Error adding {category} features: {str(e)}")
                continue
        
        # Store label
        self.labels.append(label)
        if label not in self.label_to_idx:
            self.label_to_idx[label] = len(self.label_to_idx)
        
        # Save after each ingestion to preserve data
        try:
            self.save_database('speech_database')
            print(f"Successfully added and saved sample with label: {label}")
        except Exception as e:
            print(f"Warning: Error saving database: {str(e)}")

    def find_closest_match(self, audio_path: str) -> Tuple[str, Dict[str, float]]:
        """Find the closest matching sample with detailed category scores"""
        # Extract features from input
        features = self.extract_all_features(audio_path)
        
        # Check if we have any samples in the database
        if not self.labels:
            print("No samples in database yet. Please add some samples first.")
            return "no_samples", {category: 0.0 for category in self.feature_dims.keys()}
        
        # Calculate similarity scores for each category
        category_scores = {}
        overall_confidence = 0.0
        
        for category, index in self.indices.items():
            category_features = getattr(features, f"{category}_features")
            try:
                D, I = index.search(category_features.reshape(1, -1), k=min(2, len(self.labels)))
                
                # Calculate confidence score for this category
                if len(D[0]) > 1:
                    best_distance = D[0][0]
                    second_best_distance = D[0][1]
                    confidence = 1.0 - (best_distance / (second_best_distance + 1e-6))  # Avoid division by zero
                    confidence = max(0.0, min(1.0, confidence * 1.5))
                else:
                    confidence = max(0.0, min(1.0, 1.0 - D[0][0] / 1000.0))
                
                category_scores[category] = confidence
                overall_confidence += confidence
            except Exception as e:
                print(f"Warning: Error calculating {category} score: {str(e)}")
                category_scores[category] = 0.0
        
        # Average the confidence scores
        overall_confidence /= len(self.indices)
        
        # Get the most common label from all categories
        label_votes = {}
        for category, index in self.indices.items():
            try:
                category_features = getattr(features, f"{category}_features")
                _, I = index.search(category_features.reshape(1, -1), k=1)
                voted_label = self.labels[I[0][0]]
                label_votes[voted_label] = label_votes.get(voted_label, 0) + 1
            except Exception as e:
                print(f"Warning: Error getting label for {category}: {str(e)}")
                continue
        
        if label_votes:
            closest_label = max(label_votes.items(), key=lambda x: x[1])[0]
        else:
            closest_label = "unknown"
        
        return closest_label, category_scores

    def save_database(self, path: str):
        """Save the feature database and labels"""
        # Save FAISS indices
        for category, index in self.indices.items():
            faiss.write_index(index, f"{path}_{category}_index.faiss")
        
        # Save labels and mappings
        with open(f"{path}_labels.json", 'w') as f:
            json.dump({
                'labels': self.labels,
                'label_to_idx': self.label_to_idx
            }, f)

    def load_database(self, path: str):
        """Load a saved feature database and labels"""
        # Load FAISS indices
        for category in self.feature_dims.keys():
            self.indices[category] = faiss.read_index(f"{path}_{category}_index.faiss")
        
        # Load labels and mappings
        with open(f"{path}_labels.json", 'r') as f:
            data = json.load(f)
            self.labels = data['labels']
            self.label_to_idx = data['label_to_idx']


class SpeechBrainFeatureExtractor(FeatureExtractorBase):
    """Deep learning based approach using SpeechBrain"""
    def __init__(self):
        super().__init__()
        # Initialize SpeechBrain components
        self.vad_model = None  # Lazy-load models to save startup time
        self.encoder = None
        self.mfcc_computer = MFCC(
            deltas=False,  # We'll compute deltas manually
            context=False,
            n_mfcc=40,
            n_mels=80
        )
        
    def _load_models(self):
        """Lazily load the SpeechBrain models when needed"""
        if self.vad_model is None:
            print("Loading SpeechBrain VAD model...")
            try:
                # Use run_opts to disable CUDA autocast which causes deprecation warnings
                run_opts = {"device": "cpu"}  # Use CPU to avoid CUDA warnings
                self.vad_model = VAD.from_hparams(
                    source="speechbrain/vad-crdnn-libriparty",
                    savedir="pretrained_models/vad-crdnn-libriparty",
                    run_opts=run_opts
                )
                print("VAD model loaded successfully")
            except Exception as e:
                print(f"Error loading VAD model: {str(e)}")
                # Create a placeholder that won't crash when called
                self.vad_model = type('DummyVAD', (), {'get_speech_prob': lambda self, x: torch.ones(x.shape[0])})()
                
        if self.encoder is None:
            print("Loading SpeechBrain speech encoder...")
            try:
                # Use run_opts to disable CUDA autocast which causes deprecation warnings
                run_opts = {"device": "cpu"}  # Use CPU to avoid CUDA warnings
                self.encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts=run_opts
                )
                print("Speech encoder loaded successfully")
            except Exception as e:
                print(f"Error loading speech encoder: {str(e)}")
                # Will have to fall back to traditional methods if encoder fails
    
    def extract_all_features(self, audio_path: str) -> SpeechFeatures:
        """Extract all feature categories from an audio file using SpeechBrain"""
        # Load models if not already loaded
        self._load_models()
        
        # Load audio using torchaudio (SpeechBrain works with torch tensors)
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # If stereo, convert to mono by averaging channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize waveform
        waveform = waveform / torch.max(torch.abs(waveform))
        
        try:
            # Extract all feature categories
            features = SpeechFeatures(
                rhythm_features=self.extract_rhythm_features(waveform, sr),
                energy_features=self.extract_energy_features(waveform, sr),
                pitch_features=self.extract_pitch_features(waveform, sr),
                pause_features=self.extract_pause_features(waveform, sr),
                voice_quality_features=self.extract_voice_quality_features(waveform, sr),
                speech_rate_features=self.extract_speech_rate_features(waveform, sr)
            )
            
            return features
            
        except Exception as e:
            print(f"Error in SpeechBrain feature extraction: {str(e)}")
            raise
    
    def extract_rhythm_features(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Extract rhythm features using MFCC and delta features"""
        try:
            # Compute MFCCs
            mfcc_features = self.mfcc_computer(waveform)
            
            # Compute delta and delta-delta features
            mfcc_np = mfcc_features.squeeze().numpy()
            delta_features = compute_deltas(mfcc_np)
            delta_delta_features = compute_deltas(delta_features)
            
            # Convert to numpy
            mfcc_np = mfcc_features.squeeze().numpy()
            delta_np = delta_features
            delta_delta_np = delta_delta_features
            
            # Compute statistics across time
            mfcc_mean = np.mean(mfcc_np, axis=1)
            mfcc_std = np.std(mfcc_np, axis=1)
            delta_mean = np.mean(delta_np, axis=1)
            delta_std = np.std(delta_np, axis=1)
            
            # Create feature arrays
            feature_arrays = [
                mfcc_mean[:5],  # First few MFCCs capture rhythm
                mfcc_std[:5],
                delta_mean[:5],  # Delta features capture rate of change
                delta_std[:5],
                np.mean(delta_delta_np, axis=1)[:5]  # Acceleration
            ]
            
            # Concatenate arrays
            features = np.concatenate(feature_arrays)
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['rhythm']:
                features = np.pad(features, (0, self.feature_dims['rhythm'] - len(features)))
            else:
                features = features[:self.feature_dims['rhythm']]
                
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in rhythm features: {str(e)}")
            # Return zeros as fallback
            return np.zeros(self.feature_dims['rhythm'], dtype=np.float32)
    
    def extract_energy_features(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Extract energy features using VAD probabilities and energy statistics"""
        try:
            # Convert to numpy for processing
            y_np = waveform.squeeze().numpy()
            
            # Frame-wise RMS energy (always compute this as backup)
            frame_length = int(sr * 0.025)  # 25ms frame
            hop_length = int(sr * 0.010)    # 10ms hop
            energy = np.array([
                np.sqrt(np.mean(y_np[i:i+frame_length]**2)) 
                for i in range(0, len(y_np) - frame_length, hop_length)
            ])
            
            try:
                # Try to use VAD if available
                if self.vad_model is not None:
                    speech_prob = self.vad_model.get_speech_prob(waveform)
                    speech_prob_np = speech_prob.numpy()
                    
                    # Align lengths
                    min_len = min(len(energy), len(speech_prob_np))
                    energy = energy[:min_len]
                    speech_prob_np = speech_prob_np[:min_len]
                    
                    # Weight energy by speech probability
                    weighted_energy = energy * speech_prob_np
                else:
                    # Fallback: use simple amplitude thresholding
                    threshold = np.mean(energy) * 0.1
                    speech_prob_np = (energy > threshold).astype(np.float32)
                    weighted_energy = energy
            except Exception as e:
                logging.warning(f"VAD failed, using fallback: {e}")
                # Fallback: use simple amplitude thresholding
                threshold = np.mean(energy) * 0.1
                speech_prob_np = (energy > threshold).astype(np.float32)
                weighted_energy = energy
            
            # Compute statistics
            features = np.array([
                np.mean(energy),
                np.std(energy),
                np.max(energy),
                np.mean(weighted_energy),
                np.std(weighted_energy),
                np.mean(speech_prob_np),
                skew(energy),
                kurtosis(energy)
            ])
            
            # Normalize features to be between 0 and 1
            features = np.clip(features, -100, 100)  # Clip extreme values
            features = (features - features.min()) / (features.max() - features.min() + 1e-6)
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['energy']:
                features = np.pad(features, (0, self.feature_dims['energy'] - len(features)))
            else:
                features = features[:self.feature_dims['energy']]
                
            return features.astype(np.float32)
        except Exception as e:
            logging.error(f"Error in energy features: {e}")
            raise
    
    def extract_pitch_features(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Extract pitch features using SpeechBrain encoder embeddings"""
        try:
            # Use ECAPA-TDNN embeddings which capture pitch-independent speaker traits
            embeddings = self.encoder.encode_batch(waveform)
            
            # Convert embeddings to numpy
            emb_np = embeddings.squeeze().numpy()
            
            # Get a subset of features for pitch-independent representation
            features = emb_np[:self.feature_dims['pitch']]
            
            # Pad if needed
            if len(features) < self.feature_dims['pitch']:
                features = np.pad(features, (0, self.feature_dims['pitch'] - len(features)))
                
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in pitch features: {str(e)}")
            # Fall back to zeros
            return np.zeros(self.feature_dims['pitch'], dtype=np.float32)
    
    def extract_pause_features(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Extract pause features using VAD or energy-based detection"""
        try:
            # Convert to numpy for processing
            y_np = waveform.squeeze().numpy()
            
            try:
                # Try to use VAD if available
                if self.vad_model is not None:
                    speech_prob = self.vad_model.get_speech_prob(waveform)
                    speech_prob_np = speech_prob.numpy()
                    is_speech = speech_prob_np > 0.5
                else:
                    # Fallback: use energy-based detection
                    frame_length = int(sr * 0.025)
                    hop_length = int(sr * 0.010)
                    energy = np.array([
                        np.sqrt(np.mean(y_np[i:i+frame_length]**2))
                        for i in range(0, len(y_np) - frame_length, hop_length)
                    ])
                    threshold = np.mean(energy) * 0.1
                    is_speech = energy > threshold
            except Exception as e:
                logging.warning(f"VAD failed, using energy-based detection: {e}")
                # Fallback: use energy-based detection
                frame_length = int(sr * 0.025)
                hop_length = int(sr * 0.010)
                energy = np.array([
                    np.sqrt(np.mean(y_np[i:i+frame_length]**2))
                    for i in range(0, len(y_np) - frame_length, hop_length)
                ])
                threshold = np.mean(energy) * 0.1
                is_speech = energy > threshold
            
            # Calculate pause statistics
            pause_lengths = []
            current_pause = 0
            
            for has_speech in is_speech:
                if not has_speech:
                    current_pause += 1
                elif current_pause > 0:
                    pause_lengths.append(current_pause)
                    current_pause = 0
            
            # Add final pause if needed
            if current_pause > 0:
                pause_lengths.append(current_pause)
            
            # Calculate statistics
            if pause_lengths:
                pause_lengths = np.array(pause_lengths, dtype=np.float32)
                features = np.array([
                    float(len(pause_lengths)),  # Number of pauses
                    float(np.mean(pause_lengths)),  # Average pause length
                    float(np.std(pause_lengths)) if len(pause_lengths) > 1 else 0.0,  # Pause variation
                    float(np.sum(~is_speech)) / float(len(is_speech)),  # Silence ratio
                    float(np.max(pause_lengths)) if len(pause_lengths) > 0 else 0.0,  # Longest pause
                ], dtype=np.float32)
            else:
                features = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            # Normalize features to be between 0 and 1
            if not np.all(features == 0):
                features = (features - features.min()) / (features.max() - features.min())
            
            # Pad to match expected dimension
            if len(features) < self.feature_dims['pause']:
                features = np.pad(features, (0, self.feature_dims['pause'] - len(features)))
            else:
                features = features[:self.feature_dims['pause']]
            
            return features
        except Exception as e:
            logging.error(f"Error in pause features: {e}")
            raise
    
    def extract_voice_quality_features(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Extract voice quality features using MFCC statistics"""
        try:
            # Compute MFCCs
            mfcc_features = self.mfcc_computer(waveform)
            mfcc_np = mfcc_features.squeeze().numpy()
            
            # Compute statistics
            mfcc_mean = np.mean(mfcc_np, axis=1)
            mfcc_std = np.std(mfcc_np, axis=1)
            mfcc_skew = skew(mfcc_np, axis=1)
            
            # Create feature arrays
            features = np.concatenate([
                mfcc_mean,  # Mean of MFCCs
                mfcc_std,   # Standard deviation of MFCCs
                mfcc_skew   # Skewness of MFCC distribution
            ])
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['voice_quality']:
                features = np.pad(features, (0, self.feature_dims['voice_quality'] - len(features)))
            else:
                features = features[:self.feature_dims['voice_quality']]
                
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in voice quality features: {str(e)}")
            return np.zeros(self.feature_dims['voice_quality'], dtype=np.float32)
    
    def extract_speech_rate_features(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Extract speech rate features using either MFCC or envelope-based analysis"""
        try:
            # Try MFCC-based approach first
            try:
                if self.mfcc_computer is not None:
                    mfcc_features = self.mfcc_computer(waveform)
                    mfcc_np = mfcc_features.squeeze().numpy()
                    
                    # Compute delta and delta-delta features
                    delta_features = compute_deltas(mfcc_np)
                    delta_delta_features = compute_deltas(delta_features)
                    
                    # Compute statistics
                    delta_mean = np.mean(np.abs(delta_features), axis=1)
                    delta_std = np.std(delta_features, axis=1)
                    delta_delta_mean = np.mean(np.abs(delta_delta_features), axis=1)
                    delta_delta_std = np.std(delta_delta_features, axis=1)
                    
                    # Take first 5 coefficients of each statistic
                    features = np.concatenate([
                        delta_mean[:5],      # Speed features
                        delta_std[:5],       # Speed variation
                        delta_delta_mean[:5], # Acceleration features
                        delta_delta_std[:5]   # Acceleration variation
                    ])
                else:
                    raise ValueError("MFCC computer not available")
            except Exception as e:
                logging.warning(f"MFCC-based rate extraction failed, using envelope-based: {e}")
                # Fallback: Use envelope-based analysis
                y_np = waveform.squeeze().numpy()
                
                # Compute envelope
                envelope = np.abs(hilbert(y_np))
                
                # Compute frame-wise energy
                frame_length = int(sr * 0.025)  # 25ms frames
                hop_length = int(sr * 0.010)    # 10ms hop
                frames = range(0, len(envelope) - frame_length, hop_length)
                frame_energies = np.array([
                    np.mean(envelope[i:i+frame_length]**2)
                    for i in frames
                ])
                
                # Compute rate features
                energy_diff = np.diff(frame_energies)
                energy_accel = np.diff(energy_diff)
                
                features = np.array([
                    np.mean(np.abs(energy_diff)),     # Average rate of energy change
                    np.std(energy_diff),              # Variation in rate
                    np.mean(np.abs(energy_accel)),    # Average acceleration
                    np.std(energy_accel),             # Variation in acceleration
                    np.percentile(np.abs(energy_diff), 90)  # Peak rate
                ])
                
                # Repeat features to match expected size
                features = np.tile(features, 4)[:20]
            
            # Normalize features to be between 0 and 1
            features = np.clip(features, -100, 100)  # Clip extreme values
            if not np.all(features == 0):
                features = (features - features.min()) / (features.max() - features.min())
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['speech_rate']:
                features = np.pad(features, (0, self.feature_dims['speech_rate'] - len(features)))
            else:
                features = features[:self.feature_dims['speech_rate']]
            
            return features.astype(np.float32)
        except Exception as e:
            logging.error(f"Error in speech rate features: {e}")
            raise


# Define a hybrid extractor that combines both approaches
class HybridFeatureExtractor(FeatureExtractorBase):
    """Combines the best of both traditional and deep learning approaches"""
    def __init__(self):
        super().__init__()
        # Initialize both extractors
        self.librosa_extractor = LibrosaFeatureExtractor()
        self.speechbrain_extractor = SpeechBrainFeatureExtractor()
        
    def extract_all_features(self, audio_path: str) -> SpeechFeatures:
        """Extract features using both extractors and combine them"""
        try:
            # Load audio for librosa processing
            y, sr = librosa.load(audio_path, sr=None)
            
            # Load audio for speechbrain processing
            waveform, sb_sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:  # If stereo, convert to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform / torch.max(torch.abs(waveform))
            
            # Extract features using the best method for each category
            # Rhythm: Both MFCC deltas and zero-crossing rate
            rhythm_librosa = self.librosa_extractor.extract_rhythm_features(y, sr)
            rhythm_sb = self.speechbrain_extractor.extract_rhythm_features(waveform, sb_sr)
            rhythm_features = np.concatenate([
                rhythm_librosa[:16],  # Half from librosa
                rhythm_sb[:16]        # Half from speechbrain
            ])[:self.feature_dims['rhythm']]
            
            # Energy: RMS energy + VAD-based dynamics
            energy_features = self.speechbrain_extractor.extract_energy_features(waveform, sb_sr)
            
            # Pitch: Librosa pitch tracking with robust normalization
            pitch_features = self.librosa_extractor.extract_pitch_features(y, sr)
            
            # Pause: SpeechBrain VAD for pause detection
            pause_features = self.speechbrain_extractor.extract_pause_features(waveform, sb_sr)
            
            # Voice Quality: Combined spectral and MFCC features
            voice_quality_librosa = self.librosa_extractor.extract_voice_quality_features(y, sr)
            voice_quality_sb = self.speechbrain_extractor.extract_voice_quality_features(waveform, sb_sr)
            voice_quality_features = np.concatenate([
                voice_quality_librosa[:20],  # Half from librosa
                voice_quality_sb[:20]        # Half from speechbrain
            ])[:self.feature_dims['voice_quality']]
            
            # Speech Rate: Multi-approach speech rate analysis
            speech_rate_features = self.speechbrain_extractor.extract_speech_rate_features(waveform, sb_sr)
            
            # Combine all features
            features = SpeechFeatures(
                rhythm_features=rhythm_features,
                energy_features=energy_features,
                pitch_features=pitch_features,
                pause_features=pause_features,
                voice_quality_features=voice_quality_features,
                speech_rate_features=speech_rate_features
            )
            
            return features
            
        except Exception as e:
            print(f"Error in hybrid feature extraction: {str(e)}")
            raise


# For backward compatibility, make EnhancedFeatureExtractor an alias for HybridFeatureExtractor
# This ensures existing code will use the hybrid extractor which includes all three approaches
EnhancedFeatureExtractor = HybridFeatureExtractor
