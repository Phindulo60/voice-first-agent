# Voice-First Zulu Agent — Progress Log

## What We're Building

A voice-first AI agent that lets someone speak isiZulu, get intelligent responses, and hear the answer back in isiZulu. This is the first proof-of-concept for a bigger research startup that asks: *what actually gets disadvantaged South Africans into the digital economy?*

The contrarian thesis: it's not language that's the barrier — it's infrastructure (connectivity, power, data cost). But we build the voice agent anyway because (a) it's useful, and (b) it becomes our research instrument to test that thesis.

---

## What We've Built So Far

### The Pipeline (working end-to-end)

```
[You speak Zulu] → [MMS ASR] → [NLLB Translation zu→en] → [Bedrock Claude] → [NLLB Translation en→zu] → [F5-TTS] → [You hear Zulu]
```

Every component runs locally except the LLM (Bedrock Claude), which is our only cloud dependency.

### Components

| Stage | What it does | Model | Runs where |
|-------|-------------|-------|-----------|
| **ASR** | Converts Zulu speech to text | MMS (facebook/mms-1b-all + Zulu adapter) | Local |
| **Translation (zu→en)** | Translates Zulu text to English for the LLM | NLLB-200 distilled 600M | Local |
| **LLM** | Thinks, reasons, generates response | Bedrock Claude Sonnet 4 | Cloud (AWS) |
| **Translation (en→zu)** | Translates English response back to Zulu | NLLB-200 distilled 600M | Local |
| **TTS** | Speaks the Zulu response aloud | F5-TTS v1 (flow-matching) | Local |

### What worked immediately

- **"Ngiyabonga"** → transcribed correctly → translated to "thank you" → LLM responded → translated back → spoken in Zulu. Full round trip.
- NLLB translation quality is solid for common phrases
- F5-TTS generates speech from Zulu text without crashing (on CPU)
- The whole pipeline chains together cleanly

### What we discovered (gaps)

1. **Whisper has zero Zulu support** — no Nguni languages at all in its language list. We had to swap to MMS (Meta Massively Multilingual Speech) which supports 1162 languages including Zulu.

2. **ASR accuracy degrades on longer sentences** — for "Bengifuna ukubuza umbuzo ukuthi uyakwazi na ukungisiza ukuvula i-bank account", the generic MMS adapter transcribed it with errors (`bengefunokubuza umbozo...uncetise`). Small ASR errors cascade through the whole pipeline — the translation misinterprets, the LLM gets confused, and the response is irrelevant.

3. **This validates the brief's core safety concern** — error compounds invisibly across stages, and a non-literate user can't catch it. The confirm-back / QE gate design isn't optional, it's essential.

4. **F5-TTS is slow on CPU** (~20s for short responses, ~90s for long ones) but stable. MPS (Apple Silicon GPU) causes segfaults with Zulu text.

5. **Model loading is heavy** — first run downloads ~8GB of models total (MMS 3.8GB + NLLB 2.3GB + F5-TTS 1.2GB + Whisper 150MB). After that, cached.

---

## Technical Decisions Made

| Decision | Why |
|----------|-----|
| Cascade architecture (not speech-to-speech) | No Nguni S2S model exists. Cascade gives us swappable stages + frontier reasoning |
| NLLB for translation (not in-context via LLM) | NLLB scores higher BLEU on Zulu (22.7) than GPT-4 (21.2) or Claude (19.4) |
| MMS for ASR (not Whisper) | Whisper doesn't support Zulu. MMS has dedicated Zulu adapter |
| Opus/Claude as brain (not local LLM) | InkubaLM (0.4B) can't orchestrate multi-step workflows. Need frontier reasoning |
| F5-TTS (not Polly/cloud) | Open source, runs local, supports multilingual, zero-shot voice cloning |
| Python 3.11 via uv (not system Python) | macOS libexpat mismatch on Python 3.13/3.14 breaks everything |

---

## Next Steps

### Immediate (this week)
- [ ] Get access to fine-tuned Zulu ASR model (`asr-africa/mms-1B_all_NCHLT_ZULU_50hr_v1`) — trained on 50hrs of Zulu, should handle longer sentences much better
- [ ] Test longer conversational Zulu once fine-tuned model is available
- [ ] Record a Zulu voice reference for F5-TTS (so responses sound like a Zulu speaker, not the default English voice)

### Short-term
- [ ] Add confidence scoring / back-translation check before passing to LLM (the safety layer)
- [ ] Pick a first use case (SASSA status check? airtime purchase?)
- [ ] Get primary data sources for the thesis stats (StatsSA, ICASA)
- [ ] Talk to 5 people in the target demographic

### Medium-term
- [ ] Integrate tool use / function calling (LLM actually does something, not just chat)
- [ ] WhatsApp voice note as input channel (most accessible)
- [ ] Benchmark: task completion rate + completion-without-English rate
- [ ] Write up for Deep Learning Indaba / AfricaNLP workshop

---

## How to Run It

```bash
# Clone
git clone https://github.com/Phindulo60/voice-first-agent.git
cd voice-first-agent

# Setup (Python 3.11 via uv — avoids macOS issues)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
brew install ffmpeg

# Configure AWS (for Bedrock)
ada credentials update --account 703671911115 --role admin --provider isengard --once

# Run Zulu mode
python -m src.main

# Run English mode
python -m src.main --english

# Test components individually
python -m src.asr    # Test speech recognition
python -m src.mt     # Test translation (type text)
python -m src.tts    # Test text-to-speech
python -m src.llm    # Test LLM (type text)
```

---

## Repo

**GitHub**: https://github.com/Phindulo60/voice-first-agent

---

## Research Context

This work is informed by:
- arXiv:2512.10968 — Benchmarking ASR for African Languages (W2v-BERT, MMS comparison)
- arXiv:2510.01145 — Systematic review of ASR for African low-resource languages
- arXiv:2408.17024 — InkubaLM (Lelapa AI, 0.4B African language model)
- Molefe's design brief — 9-chapter technical document covering architecture, safety, provenance, thesis

The brief's key insight: *"A voice-first Zulu agent is buildable now by borrowing the parts that are already solved and keeping native only what must be — with every risk flagged, not hidden."*

We've now proven that's true. The cascade works. The gaps are known and addressable.
