"""Configuration loaded from environment / .env file."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # AWS
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    # Bedrock
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"
    )

    # F5-TTS
    tts_device: str = os.getenv("TTS_DEVICE", "cpu")  # "cpu", "cuda", or "mps" (Apple Silicon)
    tts_ref_audio: str = os.getenv("TTS_REF_AUDIO", "")  # Path to reference voice audio
    tts_ref_text: str = os.getenv("TTS_REF_TEXT", "")  # Transcript of reference audio

    # Whisper
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")

    # Audio
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("CHANNELS", "1"))


settings = Settings()
