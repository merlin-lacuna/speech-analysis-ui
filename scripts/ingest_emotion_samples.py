import os
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from speech_feature_extractor import HybridFeatureExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingest_samples.log')
    ]
)

def validate_audio_file(file_path: Path) -> bool:
    """Validate that the audio file exists and has a supported format"""
    audio_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
    return file_path.exists() and file_path.suffix.lower() in audio_extensions

def ingest_samples(samples_dir: str, emotion_label: str, extractor: HybridFeatureExtractor) -> tuple[int, int]:
    """
    Ingest all audio samples from a directory with a given emotion label
    Returns: (number of successful ingestions, total number of samples)
    """
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        raise ValueError(f"Samples directory {samples_dir} does not exist")
    
    # Find all audio files
    audio_files = [
        f for f in samples_path.glob('**/*') 
        if validate_audio_file(f)
    ]
    
    if not audio_files:
        logging.warning(f"No valid audio files found in {samples_dir}")
        return 0, 0
    
    # Process each audio file with progress bar
    successful = 0
    total = len(audio_files)
    
    for audio_file in tqdm(audio_files, desc=f"Processing {emotion_label} samples"):
        try:
            # Try to add the sample
            extractor.add_sample(str(audio_file), emotion_label)
            successful += 1
            logging.info(f"Successfully added {audio_file} as {emotion_label}")
            
            # Save database periodically (every 10 samples)
            if successful % 10 == 0:
                try:
                    extractor.save_database('speech_database')
                    logging.info(f"Database saved after {successful} samples")
                except Exception as e:
                    logging.error(f"Error saving database: {e}")
                    
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")
            continue
    
    return successful, total

def main():
    parser = argparse.ArgumentParser(description='Ingest emotional speech samples into the database')
    parser.add_argument('--samples_dir', required=True, help='Directory containing the audio samples')
    parser.add_argument('--emotion', required=True, help='Emotion label for the samples')
    parser.add_argument('--database_path', default='speech_database', help='Path to save/load the database')
    args = parser.parse_args()
    
    try:
        # Initialize feature extractor
        logging.info("Initializing feature extractor...")
        extractor = HybridFeatureExtractor()
        
        # Load existing database if it exists
        database_files_exist = all(
            os.path.exists(f"{args.database_path}_{category}_index.faiss") 
            for category in extractor.feature_dims.keys()
        )
        
        if database_files_exist:
            logging.info("Loading existing database...")
            try:
                extractor.load_database(args.database_path)
                logging.info(f"Database loaded with {len(extractor.labels)} existing samples")
            except Exception as e:
                logging.error(f"Error loading database: {e}")
                return
        
        # Process samples
        logging.info(f"Processing samples from {args.samples_dir} as {args.emotion}...")
        successful, total = ingest_samples(args.samples_dir, args.emotion, extractor)
        
        # Save final database state
        if successful > 0:
            logging.info("Saving final database state...")
            try:
                extractor.save_database(args.database_path)
                logging.info("Database saved successfully")
            except Exception as e:
                logging.error(f"Error saving final database state: {e}")
        
        # Report results
        logging.info(f"Ingestion complete: {successful}/{total} samples processed successfully")
        if successful < total:
            logging.warning(f"Failed to process {total - successful} samples. Check the log for details.")
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 