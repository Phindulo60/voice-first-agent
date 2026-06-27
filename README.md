# Voice-First Agent

A cascaded voice agent pipeline — the foundation for a Zulu voice-first agentic system.

## Architecture

### isiZulu Pipeline (default)
```
[Mic] → [Whisper ASR] → [NLLB zu→en] → [Bedrock Claude] → [NLLB en→zu] → [F5-TTS] → [Speaker]
  local       local           local           cloud            local          local
```

### English Baseline
```
[Mic] → [Whisper ASR] → [Bedrock Claude] → [F5-TTS] → [Speaker]
```

## Setup

```bash
git clone https://github.com/Phindulo60/voice-first-agent.git
cd voice-first-agent

# Create venv with uv (avoids macOS libexpat issues)
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install FFmpeg (needed for audio processing)
brew install ffmpeg

# Configure
cp .env.example .env
# Edit .env with your AWS credentials / preferences
```

## Usage

```bash
# Zulu pipeline (default) — speak isiZulu, get isiZulu response
python -m src.main

# English-only mode
python -m src.main --english

# Test individual components
python -m src.mt           # Test translation (type Zulu/English text)
python -m src.asr          # Test speech-to-text
python -m src.llm          # Test LLM with text input
python -m src.tts          # Test text-to-speech
```

## Components

| Stage | Technology | Size | Runs |
|-------|-----------|------|------|
| ASR | faster-whisper | ~150MB | Local |
| MT | NLLB-200 (distilled 600M) | ~2.3GB | Local |
| LLM | Bedrock Claude Sonnet 4 | — | Cloud (only cloud dep) |
| TTS | F5-TTS v1 | ~1.2GB | Local |

## Notes

- First run downloads models (~4GB total). Subsequent runs use cached models.
- F5-TTS takes ~20-30s to load; once loaded, inference is fast.
- NLLB 600M is a good balance of speed and quality. For better Zulu, try `facebook/nllb-200-3.3B`.
- Whisper `base` model works for English. For better Zulu ASR, use `large-v3`.

## Voice Cloning (Optional)

Record 5-10s of clear speech, save as `ref_audio.wav`, and set in `.env`:
```
TTS_REF_AUDIO=ref_audio.wav
TTS_REF_TEXT=The exact words you spoke in the recording.
```

## Roadmap

- [x] English baseline (ASR → LLM → TTS)
- [x] isiZulu pipeline (ASR → MT → LLM → MT → TTS)
- [ ] Confidence-gated confirm-back (QE safety layer)
- [ ] Zulu-specific ASR (fine-tuned W2v-BERT)
- [ ] Zulu-specific TTS (F5-TTS fine-tuned on Zulu voice)
- [ ] Tool use / MCP integration (agentic actions)
- [ ] USSD / WhatsApp channel integration

## License

MIT
