import numpy as np
import librosa
import faiss
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import torch
import torchaudio
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert

@dataclass
class SpeechFeatures:
    rhythm_features: np.ndarray
    energy_features: np.ndarray
    pitch_features: np.ndarray
    pause_features: np.ndarray
    voice_quality_features: np.ndarray
    speech_rate_features: np.ndarray

class EnhancedFeatureExtractor:
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
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # Process MFCC deltas to ensure 1D
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            mfcc_delta_std = np.std(mfcc_delta, axis=1)
            
            # Debug print
            print("\nRhythm features shapes:", file=sys.stderr)
            print(f"zcr: {zcr.shape}", file=sys.stderr)
            print(f"tempo: {np.array([tempo]).shape}", file=sys.stderr)  # Debug tempo shape
            print(f"mfcc_delta_mean: {mfcc_delta_mean.shape}", file=sys.stderr)
            print(f"mfcc_delta_std: {mfcc_delta_std.shape}", file=sys.stderr)
            
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
                np.array([np.mean(np.abs(librosa.feature.delta(mfcc_delta_mean)))]),  # Overall rhythm change
                np.array([skew(mfcc_delta.ravel())])  # Overall skewness
            ]
            
            # Print shapes for debugging
            print("\nFeature array shapes:", file=sys.stderr)
            for i, arr in enumerate(feature_arrays):
                print(f"Array {i}: shape {arr.shape}, dims {arr.ndim}", file=sys.stderr)
            
            # Concatenate arrays
            features = np.concatenate(feature_arrays)
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['rhythm']:
                features = np.pad(features, (0, self.feature_dims['rhythm'] - len(features)))
            else:
                features = features[:self.feature_dims['rhythm']]
                
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in rhythm features: {str(e)}", file=sys.stderr)
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
            print("\nEnergy features shapes:", file=sys.stderr)
            print(f"rms: {rms.shape}", file=sys.stderr)
            print(f"spec_energy: {spec_energy.shape}", file=sys.stderr)
            
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
            print(f"Error in energy features: {str(e)}", file=sys.stderr)
            raise

    def extract_pitch_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch-independent features"""
        try:
            # Debug print
            print("\nPitch features shapes:", file=sys.stderr)
            
            # Pitch tracking
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            print(f"pitches: {pitches.shape}", file=sys.stderr)
            
            # Get pitch contour (normalized)
            pitch_contour = np.mean(pitches, axis=0)
            if len(pitch_contour) > 0:  # Check if we got any pitch values
                pitch_contour = (pitch_contour - np.mean(pitch_contour)) / (np.std(pitch_contour) + 1e-6)
            else:
                pitch_contour = np.zeros(1)  # Fallback if no pitch detected
            
            print(f"pitch_contour: {pitch_contour.shape}", file=sys.stderr)
            
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
            print(f"Error in pitch features: {str(e)}", file=sys.stderr)
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
            print("\nPause features:", file=sys.stderr)
            print(f"Number of pauses: {features[0]}", file=sys.stderr)
            print(f"Average pause length: {features[1]:.2f}", file=sys.stderr)
            print(f"Pause variation: {features[2]:.2f}", file=sys.stderr)
            print(f"Silence ratio: {features[3]:.2f}", file=sys.stderr)
            
            # Pad to match expected dimension
            if len(features) < self.feature_dims['pause']:
                features = np.pad(features, (0, self.feature_dims['pause'] - len(features)))
            else:
                features = features[:self.feature_dims['pause']]
            
            return features
        except Exception as e:
            print(f"Error in pause features: {str(e)}", file=sys.stderr)
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
            print("\nVoice quality features shapes:", file=sys.stderr)
            print(f"spectral_centroid: {spectral_centroid.shape}", file=sys.stderr)
            print(f"spectral_bandwidth: {spectral_bandwidth.shape}", file=sys.stderr)
            print(f"spectral_rolloff: {spectral_rolloff.shape}", file=sys.stderr)
            
            # MFCC-based voice quality
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            print(f"mfcc: {mfcc.shape}", file=sys.stderr)
            
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
            print(f"Error in voice quality features: {str(e)}", file=sys.stderr)
            raise

    def extract_speech_rate_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract speech rate features"""
        try:
            # Debug print
            print("\nSpeech rate features shapes:", file=sys.stderr)
            
            # Envelope-based rate analysis using scipy's hilbert
            envelope = np.abs(hilbert(y.ravel()))  # Make sure input is 1D
            print(f"envelope: {envelope.shape}", file=sys.stderr)
            
            # Reshape for RMS calculation and ensure non-empty
            envelope_frames = envelope.reshape(-1, 1)
            envelope_smooth = librosa.feature.rms(y=envelope_frames, frame_length=2048)[0]
            print(f"envelope_smooth before reshape: {envelope_smooth.shape}", file=sys.stderr)
            
            # Handle empty or invalid values
            if len(envelope_smooth) == 0 or np.any(np.isnan(envelope_smooth)):
                print("Warning: Empty or invalid envelope, using zeros", file=sys.stderr)
                # Return zero features with correct dimension
                features = np.zeros(self.feature_dims['speech_rate'], dtype=np.float32)
                return features
            
            # Ensure envelope_smooth is 1D
            envelope_smooth = envelope_smooth.ravel()
            print(f"envelope_smooth after reshape: {envelope_smooth.shape}", file=sys.stderr)
            
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
            
            print(f"Statistics: mean={env_mean}, std={env_std}, skew={env_skew}, rate={env_rate}", file=sys.stderr)
            
            # Create feature arrays one by one for debugging
            feature_arrays = [
                np.array([env_mean], dtype=np.float32),
                np.array([env_std], dtype=np.float32),
                np.array([env_skew], dtype=np.float32),
                np.array([env_rate], dtype=np.float32)
            ]
            
            # Print shapes for debugging
            print("\nSpeech rate feature array shapes:", file=sys.stderr)
            for i, arr in enumerate(feature_arrays):
                print(f"Array {i}: shape {arr.shape}, dims {arr.ndim}", file=sys.stderr)
            
            # Concatenate arrays
            features = np.concatenate(feature_arrays)
            print(f"Concatenated features: {features.shape}", file=sys.stderr)
            
            # Pad or truncate to match expected dimension
            if len(features) < self.feature_dims['speech_rate']:
                features = np.pad(features, (0, self.feature_dims['speech_rate'] - len(features)))
            else:
                features = features[:self.feature_dims['speech_rate']]
            
            print(f"Final features: {features.shape}", file=sys.stderr)
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error in speech rate features: {str(e)}", file=sys.stderr)
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
            print(f"Error in feature extraction: {str(e)}", file=sys.stderr)
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
                    print(f"Warning: NaN values in {category} features, replacing with zeros", file=sys.stderr)
                    features_reshaped = np.nan_to_num(features_reshaped, 0.0)
                index.add(features_reshaped)
            except Exception as e:
                print(f"Warning: Error adding {category} features: {str(e)}", file=sys.stderr)
                continue
        
        # Store label
        self.labels.append(label)
        if label not in self.label_to_idx:
            self.label_to_idx[label] = len(self.label_to_idx)
        
        # Save after each ingestion to preserve data
        try:
            self.save_database('speech_database')
            print(f"Successfully added and saved sample with label: {label}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error saving database: {str(e)}", file=sys.stderr)

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