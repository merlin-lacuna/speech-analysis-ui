import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from speech_feature_extractor import HybridFeatureExtractor
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import json

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

    def process_audio_file(self, audio_path: str, generate_spectrogram: bool = False, spectrogram_path: str = None) -> Dict:
        """Process an audio file and return the results"""
        try:
            # Get the match result as JSON string
            json_response = self.extractor.find_closest_match(audio_path)
            
            # Parse the JSON response
            response_data = json.loads(json_response)
            
            # Initialize result dictionary
            result = {
                "success": True,
                "logs": response_data.get("logs", "")
            }
            
            # Add match result if available
            match_result = response_data.get("result")
            if match_result:
                result["label"] = match_result["label"]
                result["confidence"] = match_result["confidence"]
            else:
                result["success"] = False
                result["error"] = "No match found"
            
            # Generate spectrogram if requested
            if generate_spectrogram and spectrogram_path:
                self._generate_spectrogram(audio_path, spectrogram_path)
                result["spectrogram_path"] = spectrogram_path
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "logs": f"Error processing audio: {str(e)}"
            }

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