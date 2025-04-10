#!/usr/bin/env python3
"""
Utility functions for speech analysis.
Provides helper functions for display and organizing test files.
"""

import os


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


def organize_test_files(file_paths):
    """Organize test files by gender and confidence level."""
    test_files = {
        "confident_male": [],
        "confident_female": [],
        "non_confident_male": [],
        "non_confident_female": []
    }
    
    for filepath in file_paths:
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
    
    return test_files