# Speech Pattern Analysis UI

An interactive art project that analyzes the users speech patterns and matches them with existing samples stored in a vector database to get an accurate label for the kind of speech pattern that the exhibit (with a focus on making it compariable regardless of the speaker's gender - since men and women speak in different pitches, we still want to be able to compare their general speech patterns without pitch differences necessarily causing a false negative.).


## Features

- Multiple feature extraction methodologies
- Pitch-independent analysis (works for any voice type)
- Real-time audio recording with silence detection
- Detailed breakdown of speech characteristics:
  - Speech Rhythm
  - Energy Control
  - Pitch Modulation
  - Pause Patterns
  - Voice Quality
  - Speech Rate
- Feature extractor comparison tools

## Requirements

```
numpy
librosa
torch
soundfile
sounddevice
faiss-cpu
sklearn
speechbrain
```


## How It Works

The system uses three different feature extraction methodologies:

### 1. Librosa-based Extractor

Traditional signal processing approach using librosa:

- Zero-crossing rate for rhythm
- RMS energy analysis
- Pitch tracking with librosa.piptrack
- Energy-based pause detection
- Spectral features for voice quality
- Hilbert envelope for speech rate

### 2. SpeechBrain-based Extractor

Deep learning based approach using SpeechBrain:

- MFCC and delta features for rhythm
- Neural VAD for energy and pauses
- SpeechBrain's pitch tracking
- MFCC statistics for voice quality
- Delta-delta features for speech rate

### 3. Hybrid Extractor

Combines the best of both approaches:

- Both MFCC deltas and zero-crossing rate for rhythm
- RMS energy + VAD-based dynamics
- Librosa pitch tracking with robust normalization
- SpeechBrain VAD for pause detection
- Combined spectral and MFCC features
- Multi-approach speech rate analysis

### Feature Categories

Each extractor analyzes six main aspects of speech:

1. Speech Rhythm:

   - Temporal patterns in speech
   - Rhythmic variation measures
   - Statistical features of speech dynamics

2. Energy Control:

   - Dynamic range analysis
   - Energy distribution patterns
   - Silence/speech transitions

3. Pitch Modulation:

   - Pitch variation patterns
   - Normalized to be pitch-independent
   - Contour statistics

4. Pause Patterns:

   - Pause frequency analysis
   - Pause duration statistics
   - Speech/silence ratio

5. Voice Quality:

   - Spectral characteristics
   - Timbre analysis
   - Voice clarity measures

6. Speech Rate:
   - Speaking speed analysis
   - Rate variation patterns
   - Temporal modulation features

### Database Components:

- Feature Storage: FAISS vector database (`faiss-cpu`)
- Feature Normalization: scikit-learn's `StandardScaler`
- Audio Processing: `soundfile` and `sounddevice`
- Numerical Operations: `numpy` for mathematical operations
- Deep Learning: `torch` and `speechbrain` for neural features

The system allows comparison between different feature extractors to analyze their strengths and differences in capturing speaking style characteristics while remaining independent of the speaker's natural pitch or voice type.
