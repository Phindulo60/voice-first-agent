"""Basic tests for the voice pipeline components."""

import pytest
import numpy as np


def test_config_loads():
    """Config loads with defaults."""
    from src.config import settings
    assert settings.aws_region == "us-east-1"
    assert settings.sample_rate == 16000


def test_whisper_model_loads():
    """Whisper model can be loaded."""
    from src.asr import get_model
    model = get_model()
    assert model is not None


def test_transcribe_silence():
    """Transcribing silence returns empty or near-empty string."""
    from src.asr import transcribe
    silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    result = transcribe(silence)
    # Silence should produce empty or very short transcription
    assert len(result.strip()) < 20


def test_llm_responds():
    """LLM returns a non-empty response."""
    from src.llm import chat
    response = chat("Hello, how are you?")
    assert len(response) > 0
    assert isinstance(response, str)


def test_tts_synthesizes():
    """TTS produces audio array."""
    from src.tts import synthesize
    audio = synthesize("Hello world")
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
