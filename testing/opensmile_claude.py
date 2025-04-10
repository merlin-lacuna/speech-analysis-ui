import opensmile
import pandas as pd

def extract_speech_features(audio_file_path):
    """Extract speech features using openSMILE eGeMAPSv02 feature set and group by metrics."""
    
    # Initialize with eGeMAPSv02 feature set
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

def print_metrics_features(results):
    """Print a readable overview of the raw speech metrics features."""
    print("\n" + "="*80)
    print(f"{'SPEECH METRICS ANALYSIS':^80}")
    print("="*80)
    
    for metric, feature_data in results.items():
        print(f"\n{metric}:")
        print("-" * 80)
        
        # Print each feature with its value and a brief description
        for column in feature_data.columns:
            value = feature_data[column].values[0]
            description = get_feature_description(column)
            print(f"  {column:<40}: {value:<12.6f} | {description}")

def get_feature_description(feature_name):
    """Return a brief description of what the feature measures."""
    descriptions = {
        # Confidence features
        'F0semitoneFrom27.5Hz_sma3nz_amean': 'Average pitch height',
        'jitterLocal_sma3nz_amean': 'Voice instability (lower is better)',
        'shimmerLocaldB_sma3nz_amean': 'Amplitude instability (lower is better)',
        'HNRdBACF_sma3nz_amean': 'Voice quality (higher is clearer)',
        'loudness_sma3_amean': 'Average loudness level',
        
        # Pitch variability features
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm': 'Pitch variation',
        'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2': 'Pitch range',
        'F0semitoneFrom27.5Hz_sma3nz_percentile20.0': 'Low pitch threshold',
        'F0semitoneFrom27.5Hz_sma3nz_percentile80.0': 'High pitch threshold',
        'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope': 'Upward pitch movement',
        'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope': 'Downward pitch movement',
        
        # Volume consistency features
        'loudness_sma3_stddevNorm': 'Volume variation',
        'loudness_sma3_percentile20.0': 'Soft volume level',
        'loudness_sma3_percentile80.0': 'Loud volume level',
        'loudness_sma3_pctlrange0-2': 'Volume range',
        'loudness_sma3_meanRisingSlope': 'Volume increase rate',
        'loudness_sma3_meanFallingSlope': 'Volume decrease rate',
        'loudnessPeaksPerSec': 'Emphasis frequency',
        
        # Articulation features
        'mfcc1_sma3_amean': 'Vocal tract shape feature 1',
        'mfcc2_sma3_amean': 'Vocal tract shape feature 2',
        'mfcc3_sma3_amean': 'Vocal tract shape feature 3',
        'mfcc4_sma3_amean': 'Vocal tract shape feature 4',
        'spectralFlux_sma3_amean': 'Speech clarity',
        'alphaRatioV_sma3nz_amean': 'Spectral energy distribution',
        'slopeV0-500_sma3nz_amean': 'Low frequency energy',
        'slopeV500-1500_sma3nz_amean': 'Mid frequency energy',
        'F1frequency_sma3nz_amean': 'First formant frequency (vowels)',
        'F2frequency_sma3nz_amean': 'Second formant frequency (vowels)',
        'F3frequency_sma3nz_amean': 'Third formant frequency (vowels)',
        
        # Strategic pausing features
        'VoicedSegmentsPerSec': 'Speaking segments per second',
        'MeanVoicedSegmentLengthSec': 'Average length of speech chunks',
        'StddevVoicedSegmentLengthSec': 'Variation in speech chunk length',
        'MeanUnvoicedSegmentLength': 'Average pause length',
        'StddevUnvoicedSegmentLength': 'Variation in pause length'
    }
    
    return descriptions.get(feature_name, "")

def analyze_speech(audio_file_path):
    """Analyze speech file and print raw feature results."""
    try:
        print(f"Analyzing speech file: {audio_file_path}")
        features, results = extract_speech_features(audio_file_path)
        print_metrics_features(results)
        
        print("\nFeature Analysis Complete!")
        print(f"Total features extracted: {len(features.columns)}")
        return features, results
    
    except Exception as e:
        print(f"Error analyzing speech: {str(e)}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Replace with your audio file path
    audio_file = "./testing/chicken.wav"
    features, results = analyze_speech(audio_file)
    
    # Optional: save full feature set to CSV for further analysis
    if features is not None:
        features.to_csv("speech_features.csv")
        print("Full feature set saved to speech_features.csv")