#!/usr/bin/env python3
"""
Utility script to build a FAISS index from a directory of audio files.
This script processes all audio files in a directory and adds them to a new FAISS index.
"""

import argparse
import os
import sys
import pathlib
import shutil
import glob

# Add the current directory to the path for module imports
script_dir = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Import from our modules
from modules.analysis import analyze_speech_comprehensive
from modules.vector_index import SpeechFeatureIndex

def build_index(audio_dir, db_path, pattern="*.wav", description_prefix="", recursive=False, new_db=False):
    """Build a FAISS index from a directory of audio files."""
    # Ensure audio_dir is an absolute path
    audio_dir = os.path.abspath(audio_dir)
    db_path = os.path.abspath(db_path)
    
    # Only delete existing database if new_db is True
    if new_db and os.path.exists(db_path):
        print(f"Creating new database: deleting existing database at {db_path}")
        shutil.rmtree(db_path)
        print(f"Previous database removed, will create a fresh index.")
    else:
        if os.path.exists(db_path):
            print(f"Updating existing database at {db_path}")
    
    # Make sure the database directory exists
    os.makedirs(db_path, exist_ok=True)
    
    # Find all audio files matching the pattern
    if recursive:
        search_pattern = os.path.join(audio_dir, "**", pattern)
        audio_files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(audio_dir, pattern)
        audio_files = glob.glob(search_pattern)
    
    # Sort files for consistent processing
    audio_files.sort()
    
    if not audio_files:
        print(f"No audio files found matching pattern {search_pattern}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Create a new index first
    index = SpeechFeatureIndex(db_path)
    
    # Process each audio file
    for i, audio_file in enumerate(audio_files):
        try:
            # Generate a description based on the filename
            filename = os.path.basename(audio_file)
            description = f"{description_prefix}{filename}" if description_prefix else filename
            
            print(f"\n[{i+1}/{len(audio_files)}] Processing {audio_file}")
            print(f"Description: {description}")
            
            # Analyze the speech file and add to the index
            analyze_speech_comprehensive(
                audio_file,
                db_path=db_path,
                description=description,
                add_to_index=True
            )
            
        except Exception as e:
            import traceback
            print(f"Error processing {audio_file}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            print("Continuing with next file...")
    
    # Get index statistics
    stats = index.get_stats()
    print("\nFAISS Index Build Complete!")
    print(f"Total speeches in index: {stats['total_speeches']}")
    print(f"Total speech segments: {stats['total_segments']}")
    print(f"Speeches with trajectory data: {stats['has_trajectory_data']}")
    print(f"Last updated: {stats['last_updated']}")

def main():
    """Parse command line arguments and build the index."""
    parser = argparse.ArgumentParser(
        description='Build a FAISS index from a directory of audio files'
    )
    parser.add_argument(
        '--audio-dir', 
        type=str, 
        required=True,
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--db', 
        type=str, 
        default='speech_features_db',
        help='Database directory for the FAISS index'
    )
    parser.add_argument(
        '--pattern', 
        type=str, 
        default="*.wav",
        help='File pattern to match (default: *.wav, supports *.mp3, etc.)'
    )
    parser.add_argument(
        '--prefix', 
        type=str, 
        default="",
        help='Optional prefix to add to descriptions'
    )
    parser.add_argument(
        '--recursive', 
        action='store_true',
        help='Search for audio files recursively in subdirectories'
    )
    parser.add_argument(
        '--new-db', 
        action='store_true',
        help='Create a new database, deleting any existing one with the same name'
    )
    
    args = parser.parse_args()
    
    build_index(
        audio_dir=args.audio_dir,
        db_path=args.db,
        pattern=args.pattern,
        description_prefix=args.prefix,
        recursive=args.recursive,
        new_db=args.new_db
    )

if __name__ == "__main__":
    main()