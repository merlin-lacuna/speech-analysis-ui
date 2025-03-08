# CEO Speech Pattern Game

An interactive art project that analyzes how well users can imitate tech CEO speech patterns (with a focus on making it accessible regardless of the speaker's natural pitch/voice type).

## Overview

This project consists of three main components:

1. `ceo_speech_ingestion.py`: Processes and stores speech samples in a vector database

   - Handles both CEO speech samples and regular speech samples
   - Supports three different feature extraction methodologies
   - Maintains a growing database of speech patterns
   - Includes tools for comparing feature extractors

2. `ceo_impression_tester.py`: Interactive "game" for testing CEO impressions

   - Records user attempts at CEO impressions
   - Compares against both CEO and regular speech patterns
   - Provides detailed feedback and scores
   - Shows which aspects of the impression need work

3. `feature_extractors/`: Three different methodologies for speech analysis
   - Librosa-based extractor (traditional signal processing)
   - SpeechBrain-based extractor (deep learning based)
   - Hybrid extractor (combines both approaches)

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

## Usage

1. First, collect audio samples:

   - CEO speech samples (e.g., Elon Musk interviews)
   - Regular speech samples (normal conversation)
   - Save them as WAV files in separate directories

2. Run the ingestion script:

   ```bash
   python ceo_speech_ingestion.py
   ```

   - Choose option 1 to add CEO samples
   - Choose option 2 to add regular speech samples
   - Option 3 to compare feature extractors
   - Add samples from both categories

3. Run the tester:
   ```bash
   python ceo_impression_tester.py
   ```
   - Choose your preferred feature extractor
   - Record your CEO impression
   - Get detailed feedback
   - Try multiple times to improve your score

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

## Art Project Context

This project explores the performative nature of tech leadership by:

- Making visible the specific speech patterns associated with tech CEOs
- Allowing anyone to practice and measure their ability to perform these patterns
- Creating an interactive experience of corporate speech performance
- Highlighting the constructed nature of "CEO-like" speech
- Comparing different methodologies for analyzing speech patterns

## Future Improvements

Potential additions:

- Add more CEO samples for variety
- Include different types of tech leaders
- Add visualization of speech patterns
- Create a web interface
- Add ability to play back matched samples
- Expand feature extractor comparison tools
- Add real-time analysis mode
