#!/usr/bin/env python3
"""
Trajectory analysis module for speech feature analysis.
Handles the analysis of how speech features change over time.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats


def analyze_trajectory(lld_features):
    """Analyze how features change over time."""
    
    # Key features to track over time
    trajectory_features = [
        'F0semitoneFrom27.5Hz_sma3nz', # Pitch
        'Loudness_sma3'                # Loudness - note correct capitalization
    ]
    
    # Initialize results dictionary
    trajectories = {}
    
    try:
        # Check if we have valid input
        if lld_features is None or lld_features.empty:
            print("Warning: Empty feature set for trajectory analysis")
            # Return a minimal set of zeros so the vector structure works
            for feature in trajectory_features:
                trajectories[f"{feature}_trend"] = 0.0
                trajectories[f"{feature}_trend_r2"] = 0.0
                trajectories[f"{feature}_shape"] = 0.0
                trajectories[f"{feature}_first"] = 0.0
                trajectories[f"{feature}_last"] = 0.0
                trajectories[f"{feature}_first_last_ratio"] = 0.0
            return trajectories
            
        # Print available columns for debugging
        print(f"Available columns for trajectory analysis: {lld_features.columns.tolist()}")
        
        # Analyze each feature trajectory
        for feature in trajectory_features:
            if feature in lld_features.columns:
                values = lld_features[feature].values
                time_index = np.arange(len(values))
                
                # Skip if no valid values (happens with pitch sometimes)
                if len(values) == 0 or np.all(np.isnan(values)):
                    print(f"Warning: No valid values for {feature}, using zeros")
                    trajectories[f"{feature}_trend"] = 0.0
                    trajectories[f"{feature}_trend_r2"] = 0.0
                    trajectories[f"{feature}_shape"] = 0.0
                    trajectories[f"{feature}_first"] = 0.0
                    trajectories[f"{feature}_last"] = 0.0
                    trajectories[f"{feature}_first_last_ratio"] = 0.0
                    continue
                    
                # Replace NaN with 0 (for pitch in unvoiced regions)
                values = np.nan_to_num(values, nan=0.0)
                    
                # Linear trend (increasing/decreasing)
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, values)
                trajectories[f"{feature}_trend"] = float(slope)
                trajectories[f"{feature}_trend_r2"] = float(r_value**2)
                
                # Shape analysis (concave/convex)
                if len(values) > 2:  # Need at least 3 points for quadratic fit
                    try:
                        coeffs = np.polyfit(time_index, values, 2)
                        trajectories[f"{feature}_shape"] = float(coeffs[0])  # a in ax² + bx + c
                    except Exception as e:
                        print(f"Error in quadratic fit for {feature}: {str(e)}")
                        trajectories[f"{feature}_shape"] = 0.0
                else:
                    trajectories[f"{feature}_shape"] = 0.0
                
                # First-last analysis
                if len(values) > 1:
                    # Use average of first/last 10% to avoid outliers
                    first_vals = values[:max(1, int(len(values) * 0.1) + 1)]
                    last_vals = values[max(0, int(len(values) * 0.9)):]
                    
                    first_mean = np.mean(first_vals)
                    last_mean = np.mean(last_vals)
                    
                    trajectories[f"{feature}_first"] = float(first_mean)
                    trajectories[f"{feature}_last"] = float(last_mean)
                    trajectories[f"{feature}_first_last_ratio"] = float(last_mean / first_mean if first_mean != 0 else 0)
                else:
                    trajectories[f"{feature}_first"] = 0.0
                    trajectories[f"{feature}_last"] = 0.0
                    trajectories[f"{feature}_first_last_ratio"] = 0.0
            else:
                print(f"Warning: Feature {feature} not found in LLD features, using zeros")
                trajectories[f"{feature}_trend"] = 0.0
                trajectories[f"{feature}_trend_r2"] = 0.0
                trajectories[f"{feature}_shape"] = 0.0
                trajectories[f"{feature}_first"] = 0.0
                trajectories[f"{feature}_last"] = 0.0
                trajectories[f"{feature}_first_last_ratio"] = 0.0
    
    except Exception as e:
        import traceback
        print(f"Error in trajectory analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return a minimal set of zeros so the vector structure works
        for feature in trajectory_features:
            trajectories[f"{feature}_trend"] = 0.0
            trajectories[f"{feature}_trend_r2"] = 0.0
            trajectories[f"{feature}_shape"] = 0.0
            trajectories[f"{feature}_first"] = 0.0
            trajectories[f"{feature}_last"] = 0.0
            trajectories[f"{feature}_first_last_ratio"] = 0.0
    
    return trajectories


def plot_trajectories(lld_features, output_dir):
    """Generate plots of key feature trajectories over time."""
    
    try:
        # Check if we have valid input
        if lld_features is None or lld_features.empty:
            print("Warning: Empty feature set for trajectory plotting")
            return output_dir
            
        # Key features to plot
        plot_features = [
            'F0semitoneFrom27.5Hz_sma3nz', # Pitch
            'Loudness_sma3'                # Loudness - note correct capitalization
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
                        try:
                            z = np.polyfit(time_axis[mask], values[mask], 1)
                            p = np.poly1d(z)
                            plt.plot(time_axis[mask], p(time_axis[mask]), "r--", alpha=0.5)
                        except Exception as e:
                            print(f"Error creating trend line for {feature}: {str(e)}")
                    
                    # Labels
                    feature_name = "Pitch" if "F0" in feature else "Loudness" if "loudness" in feature else feature
                    plt.title(f"{feature_name} over time")
                    plt.xlabel("Time (s)")
                    plt.ylabel(feature_name)
                    
                    # Save plot
                    try:
                        plot_file = os.path.join(output_dir, f"{feature}_trajectory.png")
                        plt.savefig(plot_file)
                        print(f"Saved trajectory plot to {plot_file}")
                    except Exception as e:
                        print(f"Error saving plot for {feature}: {str(e)}")
                else:
                    print(f"Warning: No valid non-NaN values for {feature}, skipping plot")
                
                plt.close()
            else:
                print(f"Warning: Feature {feature} not found in LLD features, skipping plot")
        
        return output_dir
    
    except Exception as e:
        import traceback
        print(f"Error in trajectory plotting: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return output_dir