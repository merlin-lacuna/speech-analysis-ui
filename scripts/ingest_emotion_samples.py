import os
import argparse
from speech_feature_extractor import HybridFeatureExtractor
from pathlib import Path

def ingest_samples(samples_dir: str, emotion_label: str, extractor: HybridFeatureExtractor):
    """Ingest all audio samples from a directory with a given emotion label"""
    samples_path = Path(samples_dir)
    
    # Supported audio formats
    audio_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
    
    # Process each audio file
    for audio_file in samples_path.glob('**/*'):
        if audio_file.suffix.lower() in audio_extensions:
            print(f"Processing {audio_file}...")
            try:
                extractor.add_sample(str(audio_file), emotion_label)
                print(f"Successfully added {audio_file} as {emotion_label}")
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Ingest emotional speech samples into the database')
    parser.add_argument('--samples_dir', required=True, help='Directory containing the audio samples')
    parser.add_argument('--emotion', required=True, help='Emotion label for the samples')
    parser.add_argument('--database_path', default='emotion_database', help='Path to save/load the database')
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = HybridFeatureExtractor()
    
    # Load existing database if it exists
    database_path = args.database_path
    if os.path.exists(f"{database_path}_index.faiss"):
        print("Loading existing database...")
        extractor.load_database(database_path)
    
    # Process samples
    print(f"Processing samples from {args.samples_dir} as {args.emotion}...")
    ingest_samples(args.samples_dir, args.emotion, extractor)
    
    # Save updated database
    print("Saving database...")
    extractor.save_database(database_path)
    print("Done!")

if __name__ == "__main__":
    main() 