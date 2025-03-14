import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from speech_feature_extractor import EnhancedFeatureExtractor, SpeechFeatures
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import os
import sys
import logging
import traceback

@dataclass
class AnalysisResult:
    label: str
    confidence: float
    category_scores: Dict[str, float]
    spectrogram_path: Optional[str] = None

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 22050
        self.channels = 1
        try:
            self.extractor = EnhancedFeatureExtractor()
            try:
                self.extractor.load_database('speech_database')
            except Exception as e:
                # This is expected for first-time use
                pass
        except Exception as e:
            logging.error(f"Failed to initialize feature extractor: {e}")
            raise

    def process_audio_file(self, audio_path: str, generate_spectrogram: bool = False, spectrogram_path: str = None) -> AnalysisResult:
        """Process an audio file and return analysis results"""
        try:
            # Verify audio file exists and is readable
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Verify file size
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            # Generate spectrogram if requested
            if generate_spectrogram and spectrogram_path:
                try:
                    self._generate_spectrogram(audio_path, spectrogram_path)
                except Exception as e:
                    logging.error(f"Failed to generate spectrogram: {e}")
                    raise Exception(f"Failed to generate spectrogram: {str(e)}")
            
            # Get label and category scores
            try:
                label, category_scores = self.extractor.find_closest_match(audio_path)
            except Exception as e:
                logging.error(f"Failed to analyze audio: {e}")
                raise Exception(f"Failed to analyze audio: {str(e)}")
            
            if not label or not category_scores:
                raise Exception("Analysis failed to produce valid results")
            
            # Calculate overall confidence as average of category scores
            confidence = sum(category_scores.values()) / len(category_scores)
            
            return AnalysisResult(
                label=label,
                confidence=confidence,
                category_scores=category_scores,
                spectrogram_path=spectrogram_path if generate_spectrogram else None
            )
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            raise

    def ingest_sample(self, audio_path: str, label: str):
        """Add a new sample to the database"""
        self.extractor.add_sample(audio_path, label)
        # Save after each ingestion to preserve data
        self.extractor.save_database('speech_database')

    def save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio data to a WAV file"""
        sf.write(output_path, audio, self.sample_rate)

    def _generate_spectrogram(self, audio_path: str, output_path: str):
        """Generate and save a spectrogram"""
        try:
            # Load the audio file
            try:
                y, sr = librosa.load(audio_path)
            except Exception as e:
                raise Exception(f"Failed to load audio file: {str(e)}")
            
            if len(y) == 0:
                raise Exception("Audio file contains no data")
            
            # Generate the spectrogram
            plt.figure(figsize=(10, 6))
            
            try:
                # Create mel-spectrogram for better frequency scaling
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, y_axis='linear', x_axis='time',
                                       sr=sr, cmap='viridis')
            except Exception as e:
                raise Exception(f"Failed to compute spectrogram: {str(e)}")
            
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                # Save the figure
                plt.savefig(output_path, dpi=150)
            except Exception as e:
                raise Exception(f"Failed to save spectrogram: {str(e)}")
            finally:
                plt.close()
            
            # Verify the file was created
            if not os.path.exists(output_path):
                raise Exception(f"Failed to create spectrogram at {output_path}")
                
        except Exception as e:
            raise Exception(f"Error generating spectrogram: {e}") 