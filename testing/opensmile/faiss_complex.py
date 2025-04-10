#!/usr/bin/env python3
# Speech Analysis with FAISS Indexing and Gender-Neutral Feature Matching
# This script analyzes speech using OpenSMILE and stores features in a FAISS vector database
# It supports gender-neutral feature matching for confident speech detection across genders

import opensmile
import pandas as pd
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from scipy import stats
import tempfile
import faiss
import pickle
import json
import argparse
from datetime import datetime
from pathlib import Path
from scipy.io import wavfile
from scipy.spatial.distance import cosine

class SpeechFeatureIndex:
    """Class to handle FAISS indexing of speech features."""
    
    def __init__(self, db_path="speech_features_db"):
        """Initialize the feature index."""
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Paths for different index types
        self.global_index_path = self.db_path / "global_index.faiss"
        self.segment_indices_path = self.db_path / "segment_indices.faiss"
        self.trajectory_index_path = self.db_path / "trajectory_index.faiss"
        
        # Path for metadata
        self.metadata_path = self.db_path / "metadata.pkl"
        
        # Initialize or load indices
        self.initialize_indices()
    
    def initialize_indices(self):
        """Initialize or load existing FAISS indices."""
        # Metadata dict to store mapping between indices and file paths
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {
                'global': [],      # List of (id, filepath, description) tuples
                'segments': [],    # List of (id, filepath, segment, description) tuples
                'trajectory': []   # List of (id, filepath, description) tuples
            }
        
        # Global features index
        if self.global_index_path.exists():
            self.global_index = faiss.read_index(str(self.global_index_path))
        else:
            # Create new index - dimensionality will be set when first vector is added
            self.global_index = None
        
        # Segment indices for beginning, middle, end
        if self.segment_indices_path.exists():
            self.segment_indices = faiss.read_index(str(self.segment_indices_path))
        else:
            self.segment_indices = None
        
        # Trajectory features index
        if self.trajectory_index_path.exists():
            self.trajectory_index = faiss.read_index(str(self.trajectory_index_path))
        else:
            self.trajectory_index = None
    
    def add_speech_features(self, audio_file_path, description=None):
        """Analyze speech and add features to the indices."""
        # Generate a unique ID
        speech_id = len(self.metadata['global'])
        
        # Use filename as description if none provided
        if description is None:
            description = os.path.basename(audio_file_path)
        
        # 1. Extract and add global features
        global_features, _ = extract_global_features(audio_file_path)
        global_vector = global_features.values[0].astype(np.float32)
        
        # Initialize index if this is the first entry
        if self.global_index is None:
            self.global_index = faiss.IndexFlatL2(len(global_vector))
        
        # Add to index
        self.global_index.add(global_vector.reshape(1, -1))
        
        # Add to metadata
        self.metadata['global'].append((speech_id, audio_file_path, description))
        
        # 2. Extract and add segment features
        try:
            segment_files, temp_dir = segment_audio(audio_file_path, num_segments=3)
            segment_names = ["beginning", "middle", "end"]
            
            segment_vectors = []
            for i, (segment_file, segment_name) in enumerate(zip(segment_files, segment_names)):
                segment_features, _ = extract_global_features(segment_file)
                segment_vector = segment_features.values[0].astype(np.float32)
                segment_vectors.append(segment_vector)
                
                # Add to metadata
                self.metadata['segments'].append((speech_id, audio_file_path, segment_name, description))
            
            # Combine all segment vectors into one array
            all_segments = np.vstack(segment_vectors)
            
            # Initialize segment index if this is the first entry
            if self.segment_indices is None:
                self.segment_indices = faiss.IndexFlatL2(len(segment_vectors[0]))
            
            # Add to index
            self.segment_indices.add(all_segments)
            
            # Clean up temporary files
            for file in segment_files:
                os.remove(file)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error adding segment features: {str(e)}")
        
        # 3. Extract and add trajectory features
        try:
            lld_features = extract_lld_features(audio_file_path)
            trajectories = analyze_trajectory(lld_features)
            
            # Convert to vector
            trajectory_vector = np.array(list(trajectories.values())).astype(np.float32)
            
            # Initialize trajectory index if this is the first entry
            if self.trajectory_index is None:
                self.trajectory_index = faiss.IndexFlatL2(len(trajectory_vector))
            
            # Add to index
            self.trajectory_index.add(trajectory_vector.reshape(1, -1))
            
            # Add to metadata
            self.metadata['trajectory'].append((speech_id, audio_file_path, description))
        except Exception as e:
            print(f"Error adding trajectory features: {str(e)}")
        
        # Save updated indices and metadata
        self.save_indices()
        
        return speech_id
    
    def find_similar_speeches(self, audio_file_path, k=5, gender_neutral=True, weight_config=None):
        """Find k most similar speeches to the given audio file."""
        # Extract features for the query
        global_features, _ = extract_global_features(audio_file_path)
        global_vector = global_features.values[0].astype(np.float32)
        
        if gender_neutral and weight_config:
            # Apply feature weighting
            weighted_vector = global_vector.copy()
            for idx, weight in weight_config.items():
                if idx < len(weighted_vector):
                    weighted_vector[idx] *= weight
            
            # Search with weighted vector
            D, I = self.global_index.search(weighted_vector.reshape(1, -1), k)
        else:
            # Standard search without weighting
            D, I = self.global_index.search(global_vector.reshape(1, -1), k)
        
        # Get metadata for results
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.metadata['global']):
                speech_id, filepath, desc = self.metadata['global'][idx]
                results.append({
                    'rank': i + 1,
                    'speech_id': speech_id,
                    'filepath': filepath,
                    'description': desc,
                    'similarity': 1.0 / (1.0 + D[0][i])  # Convert distance to similarity
                })
        
        return results
    
    def save_indices(self):
        """Save all indices and metadata to disk."""
        # Save indices if they exist
        if self.global_index is not None:
            faiss.write_index(self.global_index, str(self.global_index_path))
        
        if self.segment_indices is not None:
            faiss.write_index(self.segment_indices, str(self.segment_indices_path))
        
        if self.trajectory_index is not None:
            faiss.write_index(self.trajectory_index, str(self.trajectory_index_path))
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def get_stats(self):
        """Get statistics about the index."""
        stats = {
            'total_speeches': len(self.metadata['global']),
            'total_segments': len(self.metadata['segments']),
            'has_trajectory_data': len(self.metadata['trajectory']),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return stats

def print_feature_indices(audio_file_path):
    """Print the indices and names of features in the feature vector."""
    # Extract features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    features = smile.process_file(audio_file_path)
    
    # Print feature indices and names
    print("\nFEATURE INDICES FOR WEIGHTING:")
    print("=" * 80)
    for i, feature_name in enumerate(features.columns):
        feature_type = categorize_feature(feature_name)
        print(f"Index {i}: {feature_name} - Type: {feature_type}")
        
    return features.columns.tolist()

def categorize_feature(feature_name):
    """Categorize a feature by its type based on name."""
    if "F0" in feature_name:
        return "PITCH"
    elif "loudness" in feature_name:
        return "VOLUME"
    elif "jitter" in feature_name or "shimmer" in feature_name:
        return "VOICE_STABILITY"
    elif "HNR" in feature_name:
        return "VOICE_QUALITY"
    elif "mfcc" in feature_name:
        return "ARTICULATION"
    elif "Voiced" in feature_name or "Unvoiced" in feature_name:
        return "RHYTHM"
    elif "spectral" in feature_name:
        return "SPECTRAL"
    else:
        return "OTHER"

def create_feature_weights(feature_names):
    """Create a weight configuration dictionary based on feature types."""
    weights = {}
    
    # Define type-based weights for gender-neutral confidence detection
    type_weights = {
        "PITCH": 0.5,           # De-emphasize absolute pitch
        "VOICE_STABILITY": 2.0, # Emphasize jitter/shimmer (strong confidence indicators)
        "VOICE_QUALITY": 1.8,   # Emphasize voice quality features
        "RHYTHM": 1.5,          # Emphasize speaking patterns
        "VOLUME": 1.2,          # Slightly emphasize volume patterns
        "ARTICULATION": 1.3,    # Slightly emphasize articulation
        "SPECTRAL": 1.0,        # Neutral weight for spectral features
        "OTHER": 1.0            # Neutral weight for other features
    }
    
    # Special case handling for specific features
    special_weights = {
        "F0semitoneFrom27.5Hz_sma3nz_amean": 0.3,        # Strongly de-emphasize absolute pitch mean
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": 1.8,   # Emphasize pitch variation (gender-neutral)
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": 1.5, # Emphasize pitch range (gender-neutral)
        "VoicedSegmentsPerSec": 1.7                      # Emphasize speaking rate (confidence indicator)
    }
    
    # Assign weights to each feature
    for i, feature_name in enumerate(feature_names):
        # Check for special case features first
        if feature_name in special_weights:
            weights[i] = special_weights[feature_name]
        else:
            # Otherwise use the type-based weight
            feature_type = categorize_feature(feature_name)
            weights[i] = type_weights[feature_type]
    
    return weights

def modify_weights(weight_dict, modifications):
    """Modify weight configuration with new type weights."""
    new_weights = weight_dict.copy()
    
    # Get feature indices by type
    type_indices = {}
    for idx, weight in new_weights.items():
        feature_type = None
        for t, w in modifications.items():
            if t == feature_type:
                type_indices.setdefault(t, []).append(idx)
    
    # Apply modifications
    for feature_type, new_weight in modifications.items():
        if feature_type in type_indices:
            for idx in type_indices[feature_type]:
                new_weights[idx] = new_weight
    
    return new_weights

def test_weight_configuration(index, test_files, weights):
    """Test a weight configuration on a set of test files."""
    results = []
    
    # Test each confident file against all others
    for gender1 in ["male", "female"]:
        for confidence1 in ["confident", "non_confident"]:
            test_key = f"{confidence1}_{gender1}"
            if test_key in test_files and test_files[test_key]:
                test_file = test_files[test_key][0]
                
                # Search for similar speeches
                similar = index.find_similar_speeches(test_file, k=5, 
                                                     gender_neutral=True, 
                                                     weight_config=weights)
                
                # Analyze results
                for match in similar:
                    match_filename = os.path.basename(match['filepath'])
                    
                    # Determine if match is confident/non-confident and male/female
                    match_confident = "confident" in match_filename
                    match_gender = "male" if "male" in match_filename else "female"
                    
                    # Check if confidence was matched correctly across genders
                    matched_confidence = (confidence1 == "confident" and match_confident) or \
                                         (confidence1 == "non_confident" and not match_confident)
                    
                    cross_gender = gender1 != match_gender
                    
                    results.append({
                        "test_file": test_file,
                        "match_file": match['filepath'],
                        "test_confident": confidence1 == "confident",
                        "test_gender": gender1,
                        "match_confident": match_confident,
                        "match_gender": match_gender,
                        "matched_confidence_correctly": matched_confidence,
                        "cross_gender_match": cross_gender,
                        "similarity": match['similarity']
                    })
                    
                    # Print detailed result
                    print(f"  {test_file} ({confidence1} {gender1}) -> {match_filename} " +
                          f"({match_confident} {match_gender}): " +
                          f"{'✓' if matched_confidence else '✗'} " +
                          f"{'Cross-gender' if cross_gender else 'Same-gender'} " +
                          f"Similarity: {match['similarity']:.4f}")
    
    return results

def test_gender_neutral_matching(db_path="speech_features_db"):
    """Test gender-neutral matching effectiveness with different weight configurations."""
    index = SpeechFeatureIndex(db_path)
    
    # Get a list of all files in the database that have gender/confidence in the name
    all_files = []
    for speech_id, filepath, desc in index.metadata['global']:
        filename = os.path.basename(filepath)
        if any(x in filename.lower() for x in ["male", "female", "confident", "hesitant"]):
            all_files.append(filepath)
    
    if not all_files:
        print("No suitable test files found in the database. Please add files with gender/confidence indicators.")
        return None
    
    # Organize test files by gender and confidence
    test_files = {
        "confident_male": [],
        "confident_female": [],
        "non_confident_male": [],
        "non_confident_female": []
    }
    
    for filepath in all_files:
        filename = os.path.basename(filepath).lower()
        if "male" in filename:
            if "confident" in filename:
                test_files["confident_male"].append(filepath)
            elif any(x in filename for x in ["hesitant", "non_confident", "unconfident"]):
                test_files["non_confident_male"].append(filepath)
        elif "female" in filename:
            if "confident" in filename:
                test_files["confident_female"].append(filepath)
            elif any(x in filename for x in ["hesitant", "non_confident", "unconfident"]):
                test_files["non_confident_female"].append(filepath)
    
    # Make sure we have at least one file for each category
    missing_categories = [k for k, v in test_files.items() if not v]
    if missing_categories:
        print(f"Missing test files for categories: {', '.join(missing_categories)}")
        print("For best results, add files with names indicating gender and confidence level.")
        
        # Continue with available categories
        test_files = {k: v for k, v in test_files.items() if v}
    
    # Get feature names from a sample file
    sample_file = next(iter(all_files))
    feature_names = print_feature_indices(sample_file)
    
    # Define weight configurations to test
    weight_configs = {
        "equal_weights": None,  # No weighting
        "basic_gender_neutral": create_feature_weights(feature_names),
        "strong_stability": modify_weights(create_feature_weights(feature_names), 
                                          {"VOICE_STABILITY": 3.0, "PITCH": 0.2}),
        "strong_rhythm": modify_weights(create_feature_weights(feature_names),
                                       {"RHYTHM": 2.5, "PITCH": 0.2})
    }
    
    # Test each configuration
    results = {}
    for config_name, weights in weight_configs.items():
        print(f"\nTesting weight configuration: {config_name}")
        config_results = test_weight_configuration(index, test_files, weights)
        results[config_name] = config_results
        
        # Print summary statistics
        if config_results:
            correct_matches = sum(1 for r in config_results if r["matched_confidence_correctly"])
            total_tests = len(config_results)
            print(f"Accuracy: {correct_matches}/{total_tests} = {correct_matches/total_tests:.2%}")
            
            # Cross-gender accuracy
            cross_gender_results = [r for r in config_results if r["cross_gender_match"]]
            if cross_gender_results:
                cross_correct = sum(1 for r in cross_gender_results if r["matched_confidence_correctly"])
                print(f"Cross-gender accuracy: {cross_correct}/{len(cross_gender_results)} = {cross_correct/len(cross_gender_results):.2%}")
    
    # Find the best configuration
    if results:
        best_config = max(results.keys(), 
                        key=lambda c: sum(1 for r in results[c] if r["matched_confidence_correctly"]))
        
        print(f"\nRecommended weight configuration: {best_config}")
        return weight_configs[best_config]
    
    return None

def extract_global_features(audio_file_path):
    """Extract global speech features using openSMILE eGeMAPSv02 feature set with gender-neutral normalization."""
    
    # Initialize with eGeMAPSv02 feature set (global functionals)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    
    # Extract features from audio file
    features = smile.process_file(audio_file_path)
    
    # Create a new DataFrame for gender-neutral features
    gender_neutral_features = features.copy()
    
    # 1. Normalize pitch-related features to focus on patterns rather than absolute values
    # Replace absolute pitch with normalized variation metrics
    if 'F0semitoneFrom27.5Hz_sma3nz_amean' in features and 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm' in features:
        # Add coefficient of variation if it doesn't exist (stddev/mean)
        if 'F0_coefficient_of_variation' not in gender_neutral_features:
            stddev = features['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'].values[0]
            mean = features['F0semitoneFrom27.5Hz_sma3nz_amean'].values[0]
            if mean != 0:
                gender_neutral_features['F0_coefficient_of_variation'] = [stddev / mean]
            else:
                gender_neutral_features['F0_coefficient_of_variation'] = [0]
    
    # 2. Use percentile ranges instead of absolute percentiles
    if 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0' in features and 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0' in features:
        # Calculate normalized range
        p20 = features['F0semitoneFrom27.5Hz_sma3nz_percentile20.0'].values[0]
        p80 = features['F0semitoneFrom27.5Hz_sma3nz_percentile80.0'].values[0]
        
        if p20 != 0:
            gender_neutral_features['F0_percentile_ratio_80_20'] = [p80 / p20]
        else:
            gender_neutral_features['F0_percentile_ratio_80_20'] = [0]
    
    # Define feature groups based on gender-neutral features
    feature_groups = {
        'Confidence Analysis': [
            'F0semitoneFrom27.5Hz_sma3nz_amean',  # Pitch mean
            'F0_coefficient_of_variation',   # Normalized pitch variation
            'jitterLocal_sma3nz_amean',      # Voice stability
            'shimmerLocaldB_sma3nz_amean',   # Voice stability
            'HNRdBACF_sma3nz_amean',         # Voice quality
            'loudness_sma3_amean',           # Volume level
            'VoicedSegmentsPerSec'           # Speaking rate
        ],
        'Pitch Variability': [
            'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',  # Pitch variation
            'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', # Pitch range
            'F0_percentile_ratio_80_20',     # Relative pitch range
            'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', # Low pitch
            'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', # High pitch
            'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',  # Pitch rises
            'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope'  # Pitch falls
        ],
        'Volume Consistency': [
            'loudness_sma3_stddevNorm',      # Volume variation
            'loudness_sma3_percentile20.0',  # Soft volume
            'loudness_sma3_percentile80.0',  # Loud volume
            'loudness_sma3_pctlrange0-2',    # Volume range
            'loudness_sma3_meanRisingSlope', # Volume increases
            'loudness_sma3_meanFallingSlope',# Volume decreases
            'loudnessPeaksPerSec'            # Volume emphasis
        ],
        'Articulation Clarity': [
            'mfcc1_sma3_amean',              # Articulation feature
            'mfcc2_sma3_amean',              # Articulation feature
            'mfcc3_sma3_amean',              # Articulation feature
            'mfcc4_sma3_amean',              # Articulation feature
            'spectralFlux_sma3_amean',       # Speech clarity
            'alphaRatioV_sma3nz_amean',      # Spectral balance
            'slopeV0-500_sma3nz_amean',      # Spectral energy
            'slopeV500-1500_sma3nz_amean',   # Spectral energy
            'F1frequency_sma3nz_amean',      # First formant (vowel clarity)
            'F2frequency_sma3nz_amean',      # Second formant (vowel clarity)
            'F3frequency_sma3nz_amean'       # Third formant (vowel clarity)
        ],
        'Strategic Pausing': [
            'VoicedSegmentsPerSec',           # Speaking rate
            'MeanVoicedSegmentLengthSec',     # Duration of spoken segments
            'StddevVoicedSegmentLengthSec',   # Variation in speaking duration
            'MeanUnvoicedSegmentLength',      # Pause length
            'StddevUnvoicedSegmentLength'     # Variation in pause length
        ]
    }
    
    # Create a results dictionary
    results = {}
    
    # Extract and store each feature group
    for metric, feature_list in feature_groups.items():
        # Filter to only include features that exist in the features
        available_features = [f for f in feature_list if f in gender_neutral_features.columns]
        if available_features:
            results[metric] = gender_neutral_features[available_features]
    
    return gender_neutral_features, results

def extract_lld_features(audio_file_path):
    """Extract time-based low-level descriptors using openSMILE."""
    
    # Initialize with eGeMAPSv02 feature set (low-level descriptors for time analysis)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    
    # Extract features from audio file
    lld_features = smile.process_file(audio_file_path)
    
    return lld_features

def segment_audio(audio_file_path, num_segments=3):
    """Split audio into beginning, middle, and end segments."""
    
    # Load audio
    audio, sr = librosa.load(audio_file_path, sr=None)
    duration = len(audio) / sr
    
    # Calculate segment duration
    segment_duration = duration / num_segments
    
    # Create temporary directory for segments
    temp_dir = tempfile.mkdtemp()
    segment_files = []
    
    # Create segments
    for i in range(num_segments):
        start_sample = int(i * segment_duration * sr)
        end_sample = int((i + 1) * segment_duration * sr)
        
        segment = audio[start_sample:end_sample]
        
        # Save segment to temporary file
        segment_file = os.path.join(temp_dir, f"segment_{i}.wav")
        
        # Use scipy.io.wavfile instead of librosa.output.write_wav
        wavfile.write(segment_file, sr, segment.astype(np.float32))
        
        segment_files.append(segment_file)
    
    return segment_files, temp_dir

def analyze_trajectory(lld_features):
    """Analyze how features change over time."""
    
    # Key features to track over time
    trajectory_features = [
        'F0semitoneFrom27.5Hz_sma3nz', # Pitch
        'loudness_sma3'                # Loudness
    ]
    
    # Initialize results dictionary
    trajectories = {}
    
    # Analyze each feature trajectory
    for feature in trajectory_features:
        if feature in lld_features.columns:
            values = lld_features[feature].values
            time_index = np.arange(len(values))
            
            # Skip if no valid values (happens with pitch sometimes)
            if len(values) == 0 or np.all(np.isnan(values)):
                continue
                
            # Replace NaN with 0 (for pitch in unvoiced regions)
            values = np.nan_to_num(values, nan=0.0)
                
            # Linear trend (increasing/decreasing)
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, values)
            trajectories[f"{feature}_trend"] = slope
            trajectories[f"{feature}_trend_r2"] = r_value**2
            
            # Shape analysis (concave/convex)
            if len(values) > 2:  # Need at least 3 points for quadratic fit
                try:
                    coeffs = np.polyfit(time_index, values, 2)
                    trajectories[f"{feature}_shape"] = coeffs[0]  # a in ax² + bx + c
                except:
                    trajectories[f"{feature}_shape"] = 0
            
            # First-last analysis
            if len(values) > 1:
                # Use average of first/last 10% to avoid outliers
                first_vals = values[:int(len(values) * 0.1) + 1]
                last_vals = values[int(len(values) * 0.9):]
                
                first_mean = np.mean(first_vals)
                last_mean = np.mean(last_vals)
                
                trajectories[f"{feature}_first"] = first_mean
                trajectories[f"{feature}_last"] = last_mean
                trajectories[f"{feature}_first_last_ratio"] = last_mean / first_mean if first_mean != 0 else 0
    
    return trajectories

def plot_trajectories(lld_features, output_dir):
    """Generate plots of key feature trajectories over time."""
    
    # Key features to plot
    plot_features = [
        'F0semitoneFrom27.5Hz_sma3nz', # Pitch
        'loudness_sma3'                # Loudness
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each feature
    for feature in plot_features:
        if feature in lld_features.columns:
            plt.figure(figsize=(12, 6))
            
            # Get feature values
            values = lld_features[feature].values
            
            # Create time axis (avoid using DataFrame index directly)
            time_axis = np.arange(len(values)) / 100  # Assuming 100 frames per second
            
            # Handle NaN values (common in pitch)
            mask = ~np.isnan(values)
            if np.any(mask):  # Only plot if we have some valid values
                plt.plot(time_axis[mask], values[mask])
                
                # Add trend line
                if np.sum(mask) > 1:  # Need at least 2 points for a line
                    z = np.polyfit(time_axis[mask], values[mask], 1)
                    p = np.poly1d(z)
                    plt.plot(time_axis[mask], p(time_axis[mask]), "r--", alpha=0.5)
                
                # Labels
                feature_name = "Pitch" if "F0" in feature else "Loudness" if "loudness" in feature else feature
                plt.title(f"{feature_name} over time")
                plt.xlabel("Time (s)")
                plt.ylabel(feature_name)
                
                # Save plot
                plot_file = os.path.join(output_dir, f"{feature}_trajectory.png")
                plt.savefig(plot_file)
                plt.close()
    
    return os.path.join(output_dir, "trajectories")

def print_features(results, title="SPEECH METRICS ANALYSIS"):
    """Print feature groups in a readable format."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)
    
    for metric, feature_data in results.items():
        print(f"\n{metric}:")
        print("-" * 80)
        
        # Print each feature with its value
        for column in feature_data.columns:
            value = feature_data[column].values[0]
            print(f"  {column:<40}: {value:.6f}")

def analyze_speech_comprehensive(audio_file_path, db_path="speech_features_db", description=None, add_to_index=True):
    """Perform comprehensive speech analysis and add to FAISS index."""
    
    print(f"Analyzing speech file: {audio_file_path}")
    
    # 1. Global Feature Analysis
    print("\nPerforming global feature analysis...")
    global_features, global_results = extract_global_features(audio_file_path)
    
    # Print global features
    print_features(global_results, "GLOBAL SPEECH METRICS")
    
    # 2. Segmented Analysis (beginning, middle, end)
    print("\nPerforming segmented analysis (beginning, middle, end)...")
    try:
        segment_files, temp_dir = segment_audio(audio_file_path, num_segments=3)
        segment_names = ["Beginning", "Middle", "End"]
        
        segment_results = {}
        for i, (segment_file, segment_name) in enumerate(zip(segment_files, segment_names)):
            print(f"\nAnalyzing {segment_name} segment...")
            segment_features, segment_feature_groups = extract_global_features(segment_file)
            
            # Print segment features
            print_features(segment_feature_groups, f"{segment_name.upper()} SEGMENT METRICS")
            
            segment_results[segment_name.lower()] = segment_features
        
        # Clean up temporary files
        for file in segment_files:
            os.remove(file)
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error during segmented analysis: {str(e)}")
        segment_results = {}
    
    # 3. Trajectory Analysis
    print("\nPerforming trajectory analysis...")
    try:
        # Extract low-level descriptors
        lld_features = extract_lld_features(audio_file_path)
        
        # Analyze trajectories
        trajectories = analyze_trajectory(lld_features)
        
        # Create trajectory dataframe
        trajectory_df = pd.DataFrame([trajectories])
        
        # Print trajectory features
        print("\n" + "="*80)
        print(f"{'TRAJECTORY ANALYSIS':^80}")
        print("="*80)
        
        for feature, value in trajectories.items():
            print(f"  {feature:<40}: {value:.6f}")
        
        # Generate trajectory plots if requested
        try:
            output_dir = os.path.join(db_path, "plots")
            plot_dir = plot_trajectories(lld_features, output_dir)
            print(f"Trajectory plots saved to: {plot_dir}")
        except Exception as e:
            print(f"Error generating trajectory plots: {str(e)}")
        
    except Exception as e:
        print(f"Error during trajectory analysis: {str(e)}")
        trajectory_df = pd.DataFrame()
    
    # 4. Add to FAISS index if requested
    if add_to_index:
        print("\nAdding features to FAISS index...")
        try:
            index = SpeechFeatureIndex(db_path)
            speech_id = index.add_speech_features(audio_file_path, description)
            print(f"Added to index with ID: {speech_id}")
            
            # Display index statistics
            stats = index.get_stats()
            print(f"\nFAISS Index Statistics:")
            print(f"  Total speeches in index: {stats['total_speeches']}")
            print(f"  Total speech segments: {stats['total_segments']}")
            print(f"  Speeches with trajectory data: {stats['has_trajectory_data']}")
            print(f"  Last updated: {stats['last_updated']}")
        except Exception as e:
            print(f"Error adding to FAISS index: {str(e)}")
    
    print("\nComprehensive Analysis Complete!")
    
    return {
        'global_features': global_features,
        'segment_results': segment_results,
        'trajectory_features': trajectory_df
    }

def find_similar_speeches(audio_file_path, db_path="speech_features_db", k=5, gender_neutral=True):
    """Find speeches similar to the given audio file."""
    print(f"Finding speeches similar to: {audio_file_path}")
    
    index = SpeechFeatureIndex(db_path)
    
    # Create feature weights if using gender-neutral matching
    if gender_neutral:
        # Get feature names
        feature_names = extract_global_features(audio_file_path)[0].columns.tolist()
        weights = create_feature_weights(feature_names)
        print("Using gender-neutral weighting to match similar speech patterns across genders")
    else:
        weights = None
        print("Using standard similarity search")
    
    # Find similar speeches
    similar_speeches = index.find_similar_speeches(audio_file_path, k=k, 
                                                 gender_neutral=gender_neutral, 
                                                 weight_config=weights)
    
    print("\n" + "="*80)
    print(f"{'SIMILAR SPEECHES':^80}")
    print("="*80)
    
    for result in similar_speeches:
        print(f"\n{result['rank']}. {result['description']}")
        print(f"   Filepath: {result['filepath']}")
        print(f"   Similarity: {result['similarity']:.4f}")
    
    return similar_speeches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Speech Analysis with FAISS Indexing and Gender-Neutral Matching')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Add command: Analyze speech
    analyze_parser = subparsers.add_parser('analyze', help='Analyze speech and add to index')
    analyze_parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    analyze_parser.add_argument('--db', type=str, default='speech_features_db', help='Database directory')
    analyze_parser.add_argument('--description', type=str, help='Description for the speech')
    analyze_parser.add_argument('--skip-index', action='store_true', help='Skip adding to index')
    
    # Add command: Search for similar speeches
    search_parser = subparsers.add_parser('search', help='Find similar speeches')
    search_parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    search_parser.add_argument('--db', type=str, default='speech_features_db', help='Database directory')
    search_parser.add_argument('--k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--standard', action='store_true', 
                              help='Use standard search instead of gender-neutral')
    
    # Add command: Test gender-neutral matching
    test_parser = subparsers.add_parser('test', help='Test gender-neutral matching')
    test_parser.add_argument('--db', type=str, default='speech_features_db', help='Database directory')
    
    # Add command: Print feature indices
    features_parser = subparsers.add_parser('features', help='Print feature indices')
    features_parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute appropriate command
    if args.command == 'analyze':
        analyze_speech_comprehensive(
            args.file, 
            db_path=args.db, 
            description=args.description,
            add_to_index=not args.skip_index
        )
    elif args.command == 'search':
        find_similar_speeches(
            args.file, 
            db_path=args.db, 
            k=args.k, 
            gender_neutral=not args.standard
        )
    elif args.command == 'test':
        test_gender_neutral_matching(db_path=args.db)
    elif args.command == 'features':
        print_feature_indices(args.file)
    else:
        parser.print_help()