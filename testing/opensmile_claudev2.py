import opensmile
import pandas as pd
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from scipy import stats
import tempfile
from scipy.io import wavfile

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
        librosa.output.write_wav(segment_file, segment, sr)
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
            
            # Create a numerical time axis instead of using DataFrame index
            time_axis = np.arange(len(values)) / 100  # Assuming 100 frames per second, adjust if needed
            
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

def analyze_speech_comprehensive(audio_file_path, output_dir="speech_analysis_output"):
    """Perform comprehensive speech analysis with global, segmented, and trajectory features."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print(f"Analyzing speech file: {audio_file_path}")
    
    # 1. Global Feature Analysis
    print("\nPerforming global feature analysis...")
    global_features, global_results = extract_global_features(audio_file_path)
    
    # Save global features
    global_features.to_csv(os.path.join(output_dir, "global_features.csv"))
    
    # Print global features
    print_features(global_results, "GLOBAL SPEECH METRICS")
    
    results["global"] = global_features
    
    # 2. Segmented Analysis (beginning, middle, end)
    print("\nPerforming segmented analysis (beginning, middle, end)...")
    try:
        segment_files, temp_dir = segment_audio(audio_file_path, num_segments=3)
        segment_names = ["Beginning", "Middle", "End"]
        
        segment_results = {}
        for i, (segment_file, segment_name) in enumerate(zip(segment_files, segment_names)):
            print(f"\nAnalyzing {segment_name} segment...")
            segment_features, segment_feature_groups = extract_global_features(segment_file)
            
            # Save segment features
            segment_features.to_csv(os.path.join(output_dir, f"segment_{i}_{segment_name.lower()}_features.csv"))
            
            # Print segment features
            print_features(segment_feature_groups, f"{segment_name.upper()} SEGMENT METRICS")
            
            segment_results[segment_name] = segment_features
        
        results["segments"] = segment_results
        
        # Clean up temporary files
        for file in segment_files:
            os.remove(file)
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error during segmented analysis: {str(e)}")
    
    # 3. Trajectory Analysis
    print("\nPerforming trajectory analysis...")
    try:
        # Extract low-level descriptors
        lld_features = extract_lld_features(audio_file_path)
        
        # Save LLD features
        lld_features.to_csv(os.path.join(output_dir, "lld_features.csv"))
        
        # Analyze trajectories
        trajectories = analyze_trajectory(lld_features)
        
        # Create trajectory dataframe
        trajectory_df = pd.DataFrame([trajectories])
        
        # Save trajectory features
        trajectory_df.to_csv(os.path.join(output_dir, "trajectory_features.csv"))
        
        # Print trajectory features
        print("\n" + "="*80)
        print(f"{'TRAJECTORY ANALYSIS':^80}")
        print("="*80)
        
        for feature, value in trajectories.items():
            print(f"  {feature:<40}: {value:.6f}")
        
        results["trajectories"] = trajectory_df
        
        # Generate trajectory plots
        print("\nGenerating trajectory plots...")
        plot_dir = plot_trajectories(lld_features, output_dir)
        print(f"Trajectory plots saved to: {plot_dir}")
    except Exception as e:
        print(f"Error during trajectory analysis: {str(e)}")
    
    print("\nComprehensive Analysis Complete!")
    print(f"All results saved to: {output_dir}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your audio file path
    audio_file = "./testing/chicken.wav"
    results = analyze_speech_comprehensive(audio_file)