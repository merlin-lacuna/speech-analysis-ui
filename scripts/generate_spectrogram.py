import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def generate_spectrogram(audio_path, output_path):
    # Read the audio file
    sample_rate, audio_data = wavfile.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # Generate the spectrogram
    plt.figure(figsize=(10, 6))
    
    # Use scipy.signal.spectrogram for better control
    frequencies, times, Sxx = signal.spectrogram(audio_data, fs=sample_rate, 
                                                window=('hamming'), 
                                                nperseg=1024, 
                                                noverlap=512, 
                                                detrend=False, 
                                                scaling='spectrum')
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Plot the spectrogram
    plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_spectrogram.py <audio_path> <output_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        generate_spectrogram(audio_path, output_path)
        print(f"Spectrogram saved to {output_path}")
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        sys.exit(1)

