# Voice-First Agent — Baseline

English voice-in → LLM (Bedrock Claude) → English voice-out.

The simplest possible voice agent cascade. This is the foundation for the full Zulu voice-first agent — proving the pipeline works before adding translation layers.

## Architecture

```
[Mic Input] → [Whisper ASR] → [Bedrock Claude Sonnet] → [F5-TTS] → [Speaker Output]
         local                    cloud (only LLM)          local
```

## Future: Full Zulu Pipeline

```
[Mic] → [Zulu ASR] → [NLLB MT zu→en] → [Bedrock Claude] → [NLLB MT en→zu] → [F5-TTS Zulu] → [Speaker]
```

## Setup

```bash
# Clone
git clone https://github.com/Phindulo60/voice-first-agent.git
cd voice-first-agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (needed for F5-TTS audio processing)
# macOS:
brew install ffmpeg
# Ubuntu:
# sudo apt-get install ffmpeg

# Configure AWS credentials (needs Bedrock access)
export AWS_PROFILE=your-profile  # or configure ~/.aws/credentials

# Copy env template and edit
cp .env.example .env
```

## Usage

```bash
# Run the voice agent (press Enter to start/stop recording)
python -m src.main

# Test individual components
python -m src.asr          # Test speech-to-text only
python -m src.llm          # Test LLM with text input
python -m src.tts          # Test text-to-speech only
```

## Voice Cloning (Optional)

F5-TTS supports zero-shot voice cloning. To use your own voice:

1. Record a 5-10 second reference audio clip (clear speech, no background noise)
2. Save as `ref_audio.wav` in the project root
3. Set in `.env`:
```
TTS_REF_AUDIO=ref_audio.wav
TTS_REF_TEXT=The exact words spoken in your reference audio.
```

Leave blank to use F5-TTS's built-in default voice.

## Components

| Stage | Technology | Runs | Notes |
|-------|-----------|------|-------|
| ASR | faster-whisper | Local | No API call, CPU or GPU |
| LLM | Bedrock Claude Sonnet 4 | Cloud | Only cloud dependency |
| TTS | F5-TTS v1 | Local | Flow-matching, zero-shot voice cloning |

## Requirements

- Python 3.10+
- AWS account with Bedrock access
- FFmpeg installed
- Microphone
- macOS (Apple Silicon MPS) / Linux (CUDA) / CPU fallback

## License

MIT
