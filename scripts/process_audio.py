import sys
import json
import traceback
import os
from audio_processor import AudioProcessor

def process_audio(audio_path: str, spectrogram_path: str) -> dict:
    """Process audio file and return analysis results"""
    try:
        # Process audio
        processor = AudioProcessor()
        result = processor.process_audio_file(
            audio_path, 
            generate_spectrogram=True, 
            spectrogram_path=spectrogram_path
        )
        
        # Convert category scores from float32 to native Python float
        converted_scores = {
            category: float(score) 
            for category, score in result.category_scores.items()
        }
        
        # Return results as dictionary
        return {
            "emotion": result.label,
            "confidence": float(result.confidence),
            "category_scores": converted_scores
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Expect audio_path and spectrogram_path as command line arguments
    if len(sys.argv) != 3:
        print("Usage: process_audio.py <audio_path> <spectrogram_path>", file=sys.stderr)
        sys.exit(1)
        
    audio_path = sys.argv[1]
    spectrogram_path = sys.argv[2]
    
    # Print debug info to stderr
    print(f"Processing audio file: {audio_path}", file=sys.stderr)
    print(f"Spectrogram will be saved to: {spectrogram_path}", file=sys.stderr)
    
    # Process the audio and print ONLY the JSON results to stdout
    results = process_audio(audio_path, spectrogram_path)
    print(json.dumps(results)) 