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
from datetime import datetime
from pathlib import Path

def extract_global_features(audio_file_path):
    """Extract global speech features using openSMILE eGeMAPSv02 feature set."""
    
    # Initialize with eGeMAPSv02 feature set (global functionals)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    
    # Extract features from audio file
    features = smile.process_file(audio_file_path)
    
    # Define feature groups based on actual feature names in the output
    feature_groups = {
        'Confidence Analysis': [
            'F0semitoneFrom27.5Hz_sma3nz_amean',  # Pitch mean
            'jitterLocal_sma3nz_amean',           # Voice stability
            'shimmerLocaldB_sma3nz_amean',        # Voice stability
            'HNRdBACF_sma3nz_amean',              # Voice quality
            'loudness_sma3_amean',                # Volume level
            'VoicedSegmentsPerSec'                # Speaking rate
        ],
        'Pitch Variability': [
            'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',     # Pitch variation
            'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',   # Pitch range
            'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', # Low pitch
            'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', # High pitch
            'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',  # Pitch rises
            'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope'  # Pitch falls
        ],
        'Volume Consistency': [
            'loudness_sma3_stddevNorm',         # Volume variation
            'loudness_sma3_percentile20.0',     # Soft volume
            'loudness_sma3_percentile80.0',     # Loud volume
            'loudness_sma3_pctlrange0-2',       # Volume range
            'loudness_sma3_meanRisingSlope',    # Volume increases
            'loudness_sma3_meanFallingSlope',   # Volume decreases
            'loudnessPeaksPerSec'               # Volume emphasis
        ],
        'Articulation Clarity': [
            'mfcc1_sma3_amean',                 # Articulation feature
            'mfcc2_sma3_amean',                 # Articulation feature
            'mfcc3_sma3_amean',                 # Articulation feature
            'mfcc4_sma3_amean',                 # Articulation feature
            'spectralFlux_sma3_amean',          # Speech clarity
            'alphaRatioV_sma3nz_amean',         # Spectral balance
            'slopeV0-500_sma3nz_amean',         # Spectral energy
            'slopeV500-1500_sma3nz_amean',      # Spectral energy
            'F1frequency_sma3nz_amean',         # First formant (vowel clarity)
            'F2frequency_sma3nz_amean',         # Second formant (vowel clarity)
            'F3frequency_sma3nz_amean'          # Third formant (vowel clarity)
        ],
        'Strategic Pausing': [
            'VoicedSegmentsPerSec',               # Speaking rate
            'MeanVoicedSegmentLengthSec',         # Duration of spoken segments
            'StddevVoicedSegmentLengthSec',       # Variation in speaking duration
            'MeanUnvoicedSegmentLength',          # Pause length
            'StddevUnvoicedSegmentLength'         # Variation in pause length
        ]
    }
    
    # Create a results dictionary
    results = {}
    
    # Extract and store each feature group
    for metric, feature_list in feature_groups.items():
        # Filter to only include features that exist in the extracted features
        available_features = [f for f in feature_list if f in features.columns]
        results[metric] = features[available_features]
    
    return features, results

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
        from scipy.io import wavfile
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
    
    def find_similar_speeches(self, audio_file_path, k=5, description=None):
        """Find k most similar speeches to the given audio file."""
        # Extract features for the query
        global_features, _ = extract_global_features(audio_file_path)
        global_vector = global_features.values[0].astype(np.float32)
        
        # Search the global index
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

def find_similar_speeches(audio_file_path, db_path="speech_features_db", k=5):
    """Find speeches similar to the given audio file."""
    print(f"Finding speeches similar to: {audio_file_path}")
    
    index = SpeechFeatureIndex(db_path)
    similar_speeches = index.find_similar_speeches(audio_file_path, k=k)
    
    print("\n" + "="*80)
    print(f"{'SIMILAR SPEECHES':^80}")
    print("="*80)
    
    for result in similar_speeches:
        print(f"\n{result['rank']}. {result['description']}")
        print(f"   Filepath: {result['filepath']}")
        print(f"   Similarity: {result['similarity']:.4f}")
    
    return similar_speeches

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Speech Analysis with FAISS Indexing')
    parser.add_argument('--mode', type=str, choices=['analyze', 'search'], default='analyze',
                        help='Mode: analyze a speech file or search for similar speeches')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the audio file')
    parser.add_argument('--db', type=str, default='speech_features_db',
                        help='Path to the FAISS database directory')
    parser.add_argument('--description', type=str, default=None,
                        help='Optional description for the speech file')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of similar speeches to find (search mode only)')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        analyze_speech_comprehensive(args.file, args.db, args.description)
    elif args.mode == 'search':
        find_similar_speeches(args.file, args.db, args.k)