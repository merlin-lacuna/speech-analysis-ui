import sounddevice as sd
import numpy as np
import keyboard
import os
import sys
from audio_processor import AudioProcessor
import time
from threading import Event

def beep(frequency=800, duration=0.2):
    """Platform-agnostic beep function"""
    if sys.platform == 'win32':
        import winsound
        winsound.Beep(frequency, int(duration * 1000))
    else:
        # On Unix-like systems, use the terminal bell
        print('\a', end='', flush=True)

class AudioCLI:
    def __init__(self):
        self.beep_freq = 800
        self.beep_duration = 0.2  # seconds
        self.processor = AudioProcessor()
        self.stop_recording = Event()
        
        # Check audio devices
        self.devices = sd.query_devices()
        print("\nAvailable audio devices:")
        print(self.devices)
        
        # Find default input device
        try:
            self.default_input = sd.query_devices(kind='input')
            print(f"\nUsing input device: {self.default_input['name']}")
            self.device_id = self.default_input['index']
        except Exception as e:
            print(f"\nWarning: Could not get default input device: {e}")
            print("Please ensure you have a microphone connected and enabled.")
            # Try to find any input device
            for device in self.devices:
                if device['max_input_channels'] > 0:
                    print(f"Falling back to input device: {device['name']}")
                    self.device_id = device['index']
                    break
        
    def beep(self):
        """Play a beep sound"""
        try:
            beep(self.beep_freq, self.beep_duration)
        except Exception as e:
            print(f"Warning: Could not play beep sound: {e}")
    
    def wait_for_space(self):
        """Wait for space key and handle cleanup"""
        keyboard.read_event(suppress=True)  # Clear any pending events
        while True:
            event = keyboard.read_event(suppress=True)
            if event.event_type == 'down' and event.name == 'space':
                return
            elif event.event_type == 'down' and event.name == 'q':
                raise KeyboardInterrupt
    
    def record_audio(self) -> np.ndarray:
        """Record audio until space is pressed again"""
        print("\nGet ready to record...")
        print("Press SPACE when you want to start (or 'q' to quit)")
        
        try:
            self.wait_for_space()
        except KeyboardInterrupt:
            print("\nRecording cancelled")
            raise
        
        print("\n3...")
        self.beep()
        time.sleep(1)
        print("2...")
        self.beep()
        time.sleep(1)
        print("1...")
        self.beep()
        time.sleep(1)
        print("Recording NOW! (Press SPACE to stop)")
        
        # Initialize recording
        audio_data = []
        recording_error = None
        self.stop_recording.clear()
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Recording status: {status}")
            try:
                audio_data.append(indata.copy())
                # Print a dot to show recording is active
                print(".", end="", flush=True)
            except Exception as e:
                nonlocal recording_error
                recording_error = e
                print(f"\nError in recording callback: {e}")
                self.stop_recording.set()
        
        # Start recording
        try:
            with sd.InputStream(samplerate=self.processor.sample_rate, 
                              device=self.device_id,
                              channels=self.processor.channels, 
                              callback=callback,
                              dtype=np.float32) as stream:
                print("\nRecording active (each dot represents captured audio)...")
                
                # Wait for space key while recording
                try:
                    self.wait_for_space()
                except KeyboardInterrupt:
                    print("\nRecording stopped by user")
                finally:
                    self.stop_recording.set()
                
                if recording_error:
                    raise recording_error
        except Exception as e:
            print(f"\nError during recording: {e}")
            raise
        
        print("\nStopping recording...")
        self.beep()
        print("Recording stopped!")
        
        if not audio_data:
            raise RuntimeError("No audio data was recorded")
        
        # Combine all audio chunks
        try:
            audio = np.concatenate(audio_data, axis=0)
            print(f"Successfully captured audio: {audio.shape} samples")
            return audio
        except Exception as e:
            print(f"Error processing audio data: {e}")
            raise
    
    def play_audio(self, audio: np.ndarray):
        """Play back the recorded audio"""
        print("\nPlaying back recording...")
        sd.play(audio, self.processor.sample_rate)
        sd.wait()
        print("Playback complete")
    
    def ingest_mode(self):
        """Record and ingest new samples"""
        while True:
            print("\n=== INGESTION MODE ===")
            print("Record a new sample (SPACE to start/stop)")
            print("Press 'q' to quit")
            
            try:
                # Record
                audio = self.record_audio()
                
                # Playback
                self.play_audio(audio)
                
                # Ask if they want to ingest this sample
                print("\nPress 'i' to ingest this sample, any other key to discard")
                if keyboard.read_event(suppress=True).name == 'i':
                    # Get label
                    label = input("\nEnter label for this sample: ").strip()
                    
                    # Save temporary WAV file
                    temp_file = 'temp_sample.wav'
                    self.processor.save_audio(audio, temp_file)
                    
                    # Add to database
                    self.processor.ingest_sample(temp_file, label)
                    print(f"Sample ingested with label: {label}")
                    
                    # Clean up
                    os.remove(temp_file)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError during recording: {e}")
                print("You can try recording again or press 'q' to quit")
                if keyboard.read_event(suppress=True).name == 'q':
                    break

    def analyze_mode(self):
        """Record and analyze samples"""
        while True:
            print("\n=== ANALYSIS MODE ===")
            print("Record a sample to analyze (SPACE to start/stop)")
            print("Press 'q' to quit")
            
            try:
                # Record
                audio = self.record_audio()
                
                # Playback
                self.play_audio(audio)
                
                # Save temporary WAV file
                temp_file = 'temp_analysis.wav'
                self.processor.save_audio(audio, temp_file)
                
                # Analyze
                result = self.processor.process_audio_file(temp_file)
                print(f"\nAnalysis Result:")
                print(f"Best match: {result.label}")
                print(f"Confidence: {result.confidence:.2%}")
                
                # Clean up
                os.remove(temp_file)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError during recording: {e}")
                print("You can try recording again or press 'q' to quit")
                if keyboard.read_event(suppress=True).name == 'q':
                    break

def main():
    cli = AudioCLI()
    
    while True:
        print("\n=== AUDIO CLI TOOL ===")
        print("\nAvailable modes:")
        print("\n1: Ingest new samples")
        print("   - Record new audio samples and add them to the database")
        print("   - You'll be able to assign labels to your samples")
        print("   - Controls: SPACE to start/stop recording, 'i' to ingest, any other key to discard")
        print("\n2: Analyze samples")
        print("   - Record audio and compare it against your database")
        print("   - Shows the closest matching label and confidence score")
        print("   - Controls: SPACE to start/stop recording")
        print("\nq: Quit the program")
        
        choice = input("\nSelect mode (1/2/q): ")
        
        if choice == '1':
            cli.ingest_mode()
        elif choice == '2':
            cli.analyze_mode()
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid choice. Please select 1, 2, or q.")

if __name__ == "__main__":
    main() 