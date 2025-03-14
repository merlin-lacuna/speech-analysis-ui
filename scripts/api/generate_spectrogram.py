#!/usr/bin/env python3
"""
Script for generating spectrograms and analyzing audio.
Used by the generate-spectrogram API route.
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

from audio_processor import AudioProcessor

def main():
    """
    Main function to process audio and generate spectrogram.
    
    Command line arguments:
    1. audio_path: Path to the audio file to analyze
    2. output_path: Path where the spectrogram should be saved
    3. error_log_path: Path where error logs should be written (optional)
    """
    if len(sys.argv) < 3:
        print("Error: Not enough arguments. Usage: generate_spectrogram.py audio_path output_path [error_log_path]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2]
    error_log_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        # Print debugging info
        print(f"Audio file exists: {os.path.exists(audio_path)}")
        print(f"Audio file size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'file not found'}")
        
        # Process audio
        processor = AudioProcessor()
        result = processor.process_audio_file(audio_path, generate_spectrogram=True, spectrogram_path=output_path)
        
        # Convert any float32 values to regular Python floats for JSON serialization
        converted_scores = {}
        for key, value in result.category_scores.items():
            converted_scores[key] = float(value)
        
        # Print results as JSON for parsing
        print("RESULT_JSON:" + json.dumps({
            "emotion": result.label,
            "confidence": float(result.confidence),
            "category_scores": converted_scores
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