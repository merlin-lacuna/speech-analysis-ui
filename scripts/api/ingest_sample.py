#!/usr/bin/env python3
"""
Script for ingesting audio samples into the emotion database.
Used by the ingest-sample API route.
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))

from speech_feature_extractor import EnhancedFeatureExtractor

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
    
    try:
        # Print debugging info
        print(f"Scripts directory: {os.path.join(project_root, 'scripts')}")
        print(f"Database path: {database_path}")
        print(f"Audio file exists: {os.path.exists(audio_path)}")
        print(f"Audio file size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'file not found'}")
        
        # Initialize feature extractor
        extractor = EnhancedFeatureExtractor()
        
        # Check for existing database files - we need to check for one of the category indices
        try:
            if os.path.exists(f"{database_path}_rhythm_index.faiss"):
                print("Loading existing database...")
                extractor.load_database(database_path)
            else:
                print("No existing database found, starting fresh...")
        except Exception as e:
            print(f"Could not load existing database, starting fresh: {str(e)}")
        
        # Add the new sample
        print(f"Adding sample with label: {label}")
        extractor.add_sample(audio_path, label)
        
        # Save the updated database
        print("Saving database...")
        extractor.save_database(database_path)
        print("Sample successfully ingested")
        
        # Print success JSON for parsing
        print("RESULT_JSON:" + json.dumps({
            "success": True,
            "message": "Sample successfully ingested"
        }))
        
        return 0
    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        
        if error_log_path:
            with open(error_log_path, 'w') as f:
                f.write(error_msg)
        
        return 1

if __name__ == "__main__":
    sys.exit(main())