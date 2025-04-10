# Speech Analysis with FAISS Indexing

This is a modular implementation of speech analysis tools using OpenSMILE for feature extraction and FAISS for vector database indexing.

## Features

- Speech feature extraction using the OpenSMILE eGeMAPSv02 feature set
- Gender-neutral speech pattern matching
- FAISS vector database for efficient similarity search
- Speech trajectory analysis
- Comprehensive audio analysis with segmentation

## Module Structure

The codebase is organized into the following modules:

- `faiss_modular.py`: Main script with command-line interface
- `modules/`: Package containing all modular components
  - `feature_extraction.py`: Extract features from speech using OpenSMILE
  - `vector_index.py`: FAISS indexing and similarity search
  - `trajectory.py`: Analysis of how speech features change over time
  - `weights.py`: Feature weighting for gender-neutral matching
  - `utils.py`: Utility functions for display and organization
  - `analysis.py`: High-level speech analysis functions

## Usage

### Analyze a speech file

```bash
./faiss_modular.py analyze --file path/to/audio.wav
```

### Search for similar speeches

```bash
./faiss_modular.py search --file path/to/audio.wav
```

### Test gender-neutral matching

```bash
./faiss_modular.py test
```

### Print feature indices

```bash
./faiss_modular.py features --file path/to/audio.wav
```

## Requirements

- Python 3.6+
- opensmile
- pandas
- numpy
- librosa
- scipy
- matplotlib
- faiss-cpu (or faiss-gpu)