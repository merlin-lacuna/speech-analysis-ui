#!/usr/bin/env python3
"""
Speech analysis module for comprehensive speech analysis.
Provides functions to analyze speech files and find similar speeches.
"""

import os
from .feature_extraction import extract_global_features, extract_lld_features, segment_audio, print_feature_indices
from .vector_index import SpeechFeatureIndex
from .trajectory import analyze_trajectory, plot_trajectories
from .weights import create_feature_weights, modify_weights, test_weight_configuration
from .utils import print_features, organize_test_files
import pandas as pd


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
            import traceback
            print(f"Error adding to FAISS index: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
    
    print("\nComprehensive Analysis Complete!")
    
    return {
        'global_features': global_features,
        'segment_results': segment_results,
        'trajectory_features': trajectory_df
    }


def find_similar_speeches(audio_file_path, db_path="speech_features_db", k=5, gender_neutral=True, detailed=False):
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
                                                 weight_config=weights,
                                                 detailed=detailed)
    
    print("\n" + "="*80)
    print(f"{'SIMILAR SPEECHES':^80}")
    print("="*80)
    
    # Print query features if detailed mode
    if detailed and similar_speeches:
        # Get query features from first result
        if 'query_features' in similar_speeches[0] and 'query_feature_groups' in similar_speeches[0]:
            query_features = similar_speeches[0]['query_features']
            query_feature_groups = similar_speeches[0]['query_feature_groups']
            
            print("\n" + "="*80)
            print(f"{'QUERY SPEECH FEATURES':^80}")
            print("="*80)
            
            for group_name, feature_list in query_feature_groups.items():
                print(f"\n{group_name}:")
                for feature in feature_list:
                    if feature in query_features:
                        print(f"   {feature}: {query_features[feature]:.4f}")
    
    # Print results
    for result in similar_speeches:
        print(f"\n{result['rank']}. {result['description']}")
        print(f"   Filepath: {result['filepath']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        
        # If detailed, print feature group similarities
        if detailed and 'feature_group_similarities' in result:
            print(f"\n   Feature Group Similarities:")
            for group, similarity in result['feature_group_similarities'].items():
                print(f"     {group}: {similarity:.4f}")
            
            print("\n   Most Similar/Different Features:")
            if 'feature_details' in result:
                # Find top 3 most similar and different features
                feature_diffs = [(f, details['difference']) for f, details in result['feature_details'].items()]
                feature_diffs.sort(key=lambda x: x[1])
                
                # Most similar (smallest difference)
                print("     Most Similar:")
                for feature, diff in feature_diffs[:3]:
                    details = result['feature_details'][feature]
                    print(f"       {feature}: Query={details['query']:.4f}, Match={details['target']:.4f}, Diff={diff:.4f}")
                
                # Most different (largest difference)
                print("     Most Different:")
                for feature, diff in feature_diffs[-3:]:
                    details = result['feature_details'][feature]
                    print(f"       {feature}: Query={details['query']:.4f}, Match={details['target']:.4f}, Diff={diff:.4f}")
    
    return similar_speeches


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
    test_files = organize_test_files(all_files)
    
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