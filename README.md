# Voice-First Agent — Baseline

English voice-in → LLM (Bedrock Claude) → English voice-out.

The simplest possible voice agent cascade. This is the foundation for the full Zulu voice-first agent — proving the pipeline works before adding translation layers.

## Architecture

```
[Mic Input] → [Whisper ASR] → [Bedrock Claude Sonnet] → [Amazon Polly TTS] → [Speaker Output]
```

## Future: Full Zulu Pipeline

```
[Mic] → [Zulu ASR] → [MT zu→en] → [Bedrock Claude] → [MT en→zu] → [Zulu TTS] → [Speaker]
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

# Configure AWS credentials (needs Bedrock + Polly access)
export AWS_PROFILE=your-profile  # or configure ~/.aws/credentials

# Copy env template
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

## Components

| Stage | Technology | Notes |
|-------|-----------|-------|
| ASR | faster-whisper (local) | No API call, runs on CPU/GPU |
| LLM | Bedrock Claude Sonnet 4 | Via Converse API |
| TTS | Amazon Polly | Neural voices, low latency |

## Requirements

- Python 3.10+
- AWS account with Bedrock and Polly access
- Microphone
- macOS / Linux (Windows untested)

## License

MIT
