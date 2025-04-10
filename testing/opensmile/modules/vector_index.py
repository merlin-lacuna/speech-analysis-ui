#!/usr/bin/env python3
"""
FAISS vector indexing module for speech analysis.
Handles storage and retrieval of speech features using FAISS.
"""

import faiss
import pickle
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Import local modules
from .feature_extraction import extract_global_features, segment_audio, extract_lld_features
from .trajectory import analyze_trajectory


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
        
        # Check dimensions
        vector_dim = len(global_vector)
        print(f"New vector dimension: {vector_dim}")
        
        # Initialize index if this is the first entry
        if self.global_index is None:
            self.global_index = faiss.IndexFlatL2(vector_dim)
            print(f"Created new index with dimension {vector_dim}")
        else:
            # Get current index dimension
            current_dim = self.global_index.d
            print(f"Existing index dimension: {current_dim}")
            
            if current_dim != vector_dim:
                print(f"WARNING: Dimension mismatch! Index: {current_dim}, Vector: {vector_dim}")
                
                # Option 1: Create a new index with the new dimension (losing all previous data)
                # self.global_index = faiss.IndexFlatL2(vector_dim)
                # self.metadata['global'] = []
                # print(f"Created new index with dimension {vector_dim}, previous data discarded")
                
                # Option 2: Resize the vector to match the index dimension
                if vector_dim > current_dim:
                    # Truncate the vector
                    global_vector = global_vector[:current_dim]
                    print(f"Truncated vector from {vector_dim} to {current_dim} dimensions")
                else:
                    # Pad the vector with zeros
                    padded_vector = np.zeros(current_dim, dtype=np.float32)
                    padded_vector[:vector_dim] = global_vector
                    global_vector = padded_vector
                    print(f"Padded vector from {vector_dim} to {current_dim} dimensions")
        
        # Add to index
        try:
            print(f"Adding vector with shape {global_vector.reshape(1, -1).shape}")
            self.global_index.add(global_vector.reshape(1, -1))
            print("Successfully added vector to index")
        except Exception as e:
            import traceback
            print(f"Error adding to global index: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
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
            vector_dim = len(segment_vectors[0])
            print(f"Segment vector dimension: {vector_dim}")
            
            # Initialize segment index if this is the first entry
            if self.segment_indices is None:
                self.segment_indices = faiss.IndexFlatL2(vector_dim)
                print(f"Created new segment index with dimension {vector_dim}")
            else:
                # Get current index dimension
                current_dim = self.segment_indices.d
                print(f"Existing segment index dimension: {current_dim}")
                
                if current_dim != vector_dim:
                    print(f"WARNING: Segment dimension mismatch! Index: {current_dim}, Vector: {vector_dim}")
                    
                    # Fix dimension mismatch for each segment vector
                    fixed_segments = []
                    for vec in segment_vectors:
                        if len(vec) > current_dim:
                            # Truncate
                            fixed_segments.append(vec[:current_dim])
                        else:
                            # Pad
                            padded = np.zeros(current_dim, dtype=np.float32)
                            padded[:len(vec)] = vec
                            fixed_segments.append(padded)
                    
                    # Replace with fixed vectors
                    all_segments = np.vstack(fixed_segments)
                    print(f"Fixed segment vectors to match index dimension {current_dim}")
            
            # Add to index
            try:
                print(f"Adding segment vectors with shape {all_segments.shape}")
                self.segment_indices.add(all_segments)
                print("Successfully added segment vectors to index")
            except Exception as e:
                import traceback
                print(f"Error adding to segment index: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            
            # Clean up temporary files
            for file in segment_files:
                os.remove(file)
            os.rmdir(temp_dir)
        except Exception as e:
            import traceback
            print(f"Error adding segment features: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        
        # 3. Extract and add trajectory features
        try:
            lld_features = extract_lld_features(audio_file_path)
            trajectories = analyze_trajectory(lld_features)
            
            # Convert to vector
            trajectory_vector = np.array(list(trajectories.values())).astype(np.float32)
            vector_dim = len(trajectory_vector)
            print(f"Trajectory vector dimension: {vector_dim}")
            
            # Initialize trajectory index if this is the first entry
            if self.trajectory_index is None:
                self.trajectory_index = faiss.IndexFlatL2(vector_dim)
                print(f"Created new trajectory index with dimension {vector_dim}")
            else:
                # Get current index dimension
                current_dim = self.trajectory_index.d
                print(f"Existing trajectory index dimension: {current_dim}")
                
                if current_dim != vector_dim:
                    print(f"WARNING: Trajectory dimension mismatch! Index: {current_dim}, Vector: {vector_dim}")
                    
                    # Fix dimension
                    if vector_dim > current_dim:
                        # Truncate
                        trajectory_vector = trajectory_vector[:current_dim]
                        print(f"Truncated trajectory vector from {vector_dim} to {current_dim} dimensions")
                    else:
                        # Pad
                        padded = np.zeros(current_dim, dtype=np.float32)
                        padded[:vector_dim] = trajectory_vector
                        trajectory_vector = padded
                        print(f"Padded trajectory vector from {vector_dim} to {current_dim} dimensions")
            
            # Add to index
            try:
                print(f"Adding trajectory vector with shape {trajectory_vector.reshape(1, -1).shape}")
                self.trajectory_index.add(trajectory_vector.reshape(1, -1))
                print("Successfully added trajectory vector to index")
            except Exception as e:
                import traceback
                print(f"Error adding to trajectory index: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            
            # Add to metadata
            self.metadata['trajectory'].append((speech_id, audio_file_path, description))
        except Exception as e:
            import traceback
            print(f"Error adding trajectory features: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        
        # Save updated indices and metadata
        self.save_indices()
        
        return speech_id
    
    def find_similar_speeches(self, audio_file_path, k=5, gender_neutral=True, weight_config=None, detailed=False):
        """Find k most similar speeches to the given audio file."""
        # Extract features for the query
        global_features, global_feature_groups = extract_global_features(audio_file_path)
        global_vector = global_features.values[0].astype(np.float32)
        
        # Store query features for detailed comparison
        query_features = global_features.iloc[0].to_dict()
        query_feature_groups = global_feature_groups
        
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
                
                result_info = {
                    'rank': i + 1,
                    'speech_id': speech_id,
                    'filepath': filepath,
                    'description': desc,
                    'similarity': 1.0 / (1.0 + D[0][i] * 0.01)  # Scale distance for better visibility
                }
                
                # If detailed is requested, add feature group comparisons
                if detailed:
                    # Extract target speech features
                    target_features, target_feature_groups = extract_global_features(filepath)
                    
                    # Calculate feature group similarities
                    feature_similarities = {}
                    feature_group_similarities = {}
                    
                    # Feature group comparison
                    for group_name, features in query_feature_groups.items():
                        # Skip if group not present in both
                        if group_name not in target_feature_groups:
                            continue
                            
                        # Get feature values for this group from both speeches
                        query_group_features = {k: query_features[k] for k in features if k in query_features}
                        target_group_features = {k: target_features.iloc[0][k] for k in features if k in target_features.columns}
                        
                        # Calculate group similarity
                        if query_group_features and target_group_features:
                            # Common features
                            common_features = set(query_group_features.keys()) & set(target_group_features.keys())
                            
                            if common_features:
                                # Vector comparison for this group
                                query_vector = np.array([query_group_features[k] for k in common_features])
                                target_vector = np.array([target_group_features[k] for k in common_features])
                                
                                # Calculate distance and convert to similarity
                                dist = np.linalg.norm(query_vector - target_vector)
                                group_similarity = 1.0 / (1.0 + dist * 0.01)
                                
                                feature_group_similarities[group_name] = group_similarity
                                
                                # Store individual feature values for comparison
                                for feature in common_features:
                                    feature_similarities[feature] = {
                                        'query': query_group_features[feature],
                                        'target': target_group_features[feature],
                                        'difference': abs(query_group_features[feature] - target_group_features[feature])
                                    }
                    
                    # Add the detailed information
                    result_info['feature_group_similarities'] = feature_group_similarities
                    result_info['feature_details'] = feature_similarities
                    
                    # Add query features for reference
                    if i == 0:  # Only need to add this once
                        result_info['query_feature_groups'] = query_feature_groups
                        result_info['query_features'] = query_features
                
                results.append(result_info)
        
        return results
    
    def save_indices(self):
        """Save all indices and metadata to disk."""
        try:
            # Ensure the directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Save indices if they exist
            if self.global_index is not None:
                faiss.write_index(self.global_index, str(self.global_index_path))
                print(f"Global index saved to {self.global_index_path}")
            
            if self.segment_indices is not None:
                faiss.write_index(self.segment_indices, str(self.segment_indices_path))
                print(f"Segment indices saved to {self.segment_indices_path}")
            
            if self.trajectory_index is not None:
                faiss.write_index(self.trajectory_index, str(self.trajectory_index_path))
                print(f"Trajectory index saved to {self.trajectory_index_path}")
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
                print(f"Metadata saved to {self.metadata_path}")
                
        except Exception as e:
            import traceback
            print(f"Error saving indices: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
    
    def get_stats(self):
        """Get statistics about the index."""
        stats = {
            'total_speeches': len(self.metadata['global']),
            'total_segments': len(self.metadata['segments']),
            'has_trajectory_data': len(self.metadata['trajectory']),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return stats