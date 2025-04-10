#!/usr/bin/env python3
"""
Feature extraction module for speech analysis.
Handles extraction of global features, low-level descriptors,
and audio segmentation.
"""

import opensmile
import pandas as pd
import numpy as np
import librosa
import os
import tempfile
from scipy.io import wavfile


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