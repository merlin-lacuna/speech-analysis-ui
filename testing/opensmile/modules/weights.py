#!/usr/bin/env python3
"""
Feature weighting module for gender-neutral speech analysis.
Provides functions to create and modify weight configurations
for gender-neutral speech analysis.
"""

import os
from .feature_extraction import categorize_feature


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
    
    try:
        # Assign weights to each feature
        for i, feature_name in enumerate(feature_names):
            # Check for special case features first
            if feature_name in special_weights:
                weights[i] = special_weights[feature_name]
            else:
                # Otherwise use the type-based weight
                feature_type = categorize_feature(feature_name)
                weights[i] = type_weights[feature_type]
    except Exception as e:
        import traceback
        print(f"Error creating feature weights: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return a default empty weights dict if we can't create weights
        return {}
    
    return weights


def modify_weights(weight_dict, modifications):
    """Modify weight configuration with new type weights."""
    if not weight_dict:
        return {}
        
    try:
        new_weights = weight_dict.copy()
        
        # Since we don't have access to original feature names here,
        # we'll simply apply any modifications that match a type directly
        # This is a simplified version that doesn't require feature_type mapping
        
        # Apply modifications to existing weights that match the type pattern
        # This is a placeholder implementation - actual modification would need feature names
        for idx, current_weight in new_weights.items():
            # Apply a simple heuristic - if the current weight is in the range of a specific type
            # For example, PITCH weights are typically 0.5, VOICE_STABILITY around 2.0, etc.
            if current_weight <= 0.6 and "PITCH" in modifications:
                new_weights[idx] = modifications["PITCH"]
            elif 1.9 <= current_weight <= 2.1 and "VOICE_STABILITY" in modifications:
                new_weights[idx] = modifications["VOICE_STABILITY"]
            elif 1.4 <= current_weight <= 1.6 and "RHYTHM" in modifications:
                new_weights[idx] = modifications["RHYTHM"]
        
        return new_weights
    except Exception as e:
        import traceback
        print(f"Error modifying weights: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return weight_dict  # Return original dictionary if modification fails


def test_weight_configuration(index, test_files, weights):
    """Test a weight configuration on a set of test files."""
    results = []
    
    # Test each confident file against all others
    for gender1 in ["male", "female"]:
        for confidence1 in ["confident", "non_confident"]:
            test_key = f"{confidence1}_{gender1}"
            if test_key in test_files and test_files[test_key]:
                test_file = test_files[test_key][0]
                
                try:
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
                except Exception as e:
                    import traceback
                    print(f"Error testing configuration with file {test_file}: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
    
    return results