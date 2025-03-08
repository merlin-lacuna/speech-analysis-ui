import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from speech_feature_extractor import HybridFeatureExtractor
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    label: str
    confidence: float
    features: Dict[str, float]
    spectrogram_path: Optional[str] = None

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 22050
        self.channels = 1
        self.extractor = HybridFeatureExtractor()
        self.extractor.load_database('emotion_database')

    def process_audio_file(self, audio_path: str, generate_spectrogram: bool = False, spectrogram_path: str = None) -> AnalysisResult:
        """Process an audio file and return analysis results"""
        # Generate spectrogram if requested
        if generate_spectrogram and spectrogram_path:
            self._generate_spectrogram(audio_path, spectrogram_path)
        
        # Get label and confidence
        label, confidence = self.extractor.find_closest_match(audio_path)
        
        # Get features for debugging/display
        features = self.extractor.extract_all_features(audio_path)
        feature_dict = {
            f"feature_{i}": float(val) for i, val in enumerate(features)
        }
        
        return AnalysisResult(
            label=label,
            confidence=confidence,
            features=feature_dict,
            spectrogram_path=spectrogram_path if generate_spectrogram else None
        )

    def ingest_sample(self, audio_path: str, label: str):
        """Add a new sample to the database"""
        self.extractor.add_sample(audio_path, label)

    def save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio data to a WAV file"""
        sf.write(output_path, audio, self.sample_rate)

    def _generate_spectrogram(self, audio_path: str, output_path: str):
        """Generate and save a spectrogram"""
        # Load the audio file
        y, sr = librosa.load(audio_path)
        
        # Generate the spectrogram
        plt.figure(figsize=(10, 6))
        
        # Create mel-spectrogram for better frequency scaling
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=sr, cmap='viridis')
        
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=150)
        plt.close() 