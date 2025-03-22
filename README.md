# Audio Spectrogram Generator

A tool for generating spectrograms from audio files and analyzing emotions.

## Requirements

### System Requirements

- Python 3.8 or higher
- Node.js 16 or higher
- FFmpeg installed on your system
- On Linux systems, you may need to run the CLI tool with sudo privileges for keyboard input handling

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Node.js Dependencies

Install the required Node.js packages:

```bash
npm install
```

## Running the Application

1. Start the development server:

```bash
npm run dev
```

2. Open your browser and navigate to `http://localhost:3000`

## CLI Tool Usage

The CLI tool can be used for recording and analyzing audio samples directly from the command line.

### Linux Users

If you encounter permission issues with keyboard input on Linux, you may need to run the CLI tool with sudo:

```bash
sudo python scripts/cli_tool.py
```

### Available Modes

1. Ingest Mode: Record and add new samples to the database
2. Analysis Mode: Record and analyze audio samples against the database

## Platform Support

This application is designed to work on both Windows and Linux systems. The CLI tool uses platform-specific features for audio feedback and keyboard input handling.
