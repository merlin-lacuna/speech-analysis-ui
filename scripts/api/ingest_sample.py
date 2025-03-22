#!/usr/bin/env python3
"""
Script for ingesting audio samples into the emotion database.
Used by the ingest-sample API route.
"""

import sys
import os
import json
import logging
import traceback
import shutil
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))

from speech_feature_extractor import EnhancedFeatureExtractor

def setup_logging(error_log_path=None):
    """Set up logging configuration"""
    handlers = [logging.StreamHandler()]
    if error_log_path:
        handlers.append(logging.FileHandler(error_log_path))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def validate_audio_file(audio_path: str) -> bool:
    """Validate that the audio file exists and has content"""
    if not os.path.exists(audio_path):
        logging.error(f"Audio file does not exist: {audio_path}")
        return False
    
    try:
        size = os.path.getsize(audio_path)
        if size == 0:
            logging.error(f"Audio file is empty: {audio_path}")
            return False
        logging.info(f"Audio file validated: {audio_path} ({size} bytes)")
        return True
    except Exception as e:
        logging.error(f"Error validating audio file: {e}")
        return False

def ensure_training_dir(label: str) -> str:
    """Ensure the training samples directory exists for the given label"""
    public_dir = os.path.join(project_root, 'public')
    training_dir = os.path.join(public_dir, 'training_samples', label)
    os.makedirs(training_dir, exist_ok=True)
    return training_dir

def move_to_training_dir(audio_path: str, label: str) -> str:
    """Move the audio file to the appropriate training samples directory"""
    training_dir = ensure_training_dir(label)
    filename = os.path.basename(audio_path)
    new_path = os.path.join(training_dir, filename)
    
    # If file already exists in training dir, create a unique name
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(new_path):
        new_path = os.path.join(training_dir, f"{base}_{counter}{ext}")
        counter += 1
    
    shutil.copy2(audio_path, new_path)  # Copy with metadata
    logging.info(f"Copied audio file to training directory: {new_path}")
    return new_path

def load_or_create_database(extractor: EnhancedFeatureExtractor, database_path: str) -> bool:
    """Load existing database or create new one"""
    try:
        # Check for all required index files
        indices_exist = all(
            os.path.exists(f"{database_path}_{category}_index.faiss") 
            for category in extractor.feature_dims.keys()
        )
        
        if indices_exist:
            logging.info("Loading existing database...")
            extractor.load_database(database_path)
            logging.info(f"Database loaded with {len(extractor.labels)} existing samples")
            return True
        else:
            logging.info("No complete database found, starting fresh...")
            return True
    except Exception as e:
        logging.error(f"Error loading database: {e}")
        return False

def main():
    """
    Main function to ingest an audio sample into the database.
    
    Command line arguments:
    1. audio_path: Path to the audio file to ingest
    2. label: Emotion label for the audio sample
    3. database_path: Path to the emotion database
    4. error_log_path: Path where error logs should be written (optional)
    """
    if len(sys.argv) < 4:
        print("Error: Not enough arguments. Usage: ingest_sample.py audio_path label database_path [error_log_path]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    label = sys.argv[2]
    database_path = sys.argv[3]
    error_log_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Set up logging
    setup_logging(error_log_path)
    
    try:
        # Validate inputs
        if not validate_audio_file(audio_path):
            raise ValueError("Invalid audio file")
        
        if not label.strip():
            raise ValueError("Label cannot be empty")
        
        # Move the audio file to the training samples directory
        new_audio_path = move_to_training_dir(audio_path, label)
        
        # Initialize feature extractor
        logging.info("Initializing feature extractor...")
        extractor = EnhancedFeatureExtractor()
        
        # Load or create database
        if not load_or_create_database(extractor, database_path):
            raise RuntimeError("Failed to initialize database")
        
        # Add the new sample using the new path
        logging.info(f"Adding sample with label: {label}")
        try:
            extractor.add_sample(new_audio_path, label)
        except Exception as e:
            logging.error(f"Failed to add sample: {e}")
            # Check if any features were extracted successfully
            if hasattr(e, 'partial_success') and e.partial_success:
                logging.warning("Some features were extracted but not all")
            raise
        
        # Save the updated database
        logging.info("Saving database...")
        try:
            extractor.save_database(database_path)
            logging.info("Database saved successfully")
        except Exception as e:
            logging.error(f"Failed to save database: {e}")
            raise
        
        # Return detailed success response
        result = {
            "success": True,
            "message": "Sample successfully ingested",
            "details": {
                "database_size": len(extractor.labels),
                "label": label,
                "audio_path": f"/training_samples/{label}/{os.path.basename(new_audio_path)}"
            }
        }
        
        print("RESULT_JSON:" + json.dumps(result))
        return 0
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        # Check if this is just a deprecation warning about torch.cuda.amp.custom_fwd
        # but the sample was actually successfully added
        is_only_deprecation_warning = (
            'torch.cuda.amp.custom_fwd' in error_traceback and
            'Database saved successfully' in error_traceback and
            not any(err in error_traceback for err in ['ImportError', 'ValueError', 'FileNotFoundError'])
        )
        
        if is_only_deprecation_warning:
            # If it's just the deprecation warning but the sample was added, return success
            logging.warning("Ignoring SpeechBrain deprecation warning")
            result = {
                "success": True,
                "message": "Sample successfully ingested (with warnings)",
                "details": {
                    "label": label,
                    "audio_path": f"/training_samples/{label}/{os.path.basename(new_audio_path)}",
                    "warnings": True
                }
            }
            print("RESULT_JSON:" + json.dumps(result))
            return 0
            
        # Log actual errors
        full_error_msg = f"Error: {error_msg}\nTraceback:\n{error_traceback}"
        if not ('torch.cuda.amp.custom_fwd' in full_error_msg and 'FutureWarning' in full_error_msg):
            logging.error(full_error_msg)
        
        if error_log_path:
            with open(error_log_path, 'w') as f:
                f.write(full_error_msg)
        
        # Return structured error response
        error_result = {
            "success": False,
            "error": error_msg,
            "details": {
                "type": e.__class__.__name__,
                "message": error_msg,
                "traceback": error_traceback
            }
        }
        print("RESULT_JSON:" + json.dumps(error_result))
        return 1

if __name__ == "__main__":
    sys.exit(main())