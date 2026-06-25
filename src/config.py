"""Configuration loaded from environment / .env file."""

import os
from dataclasses import dataclass
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

    # Polly
    polly_voice_id: str = os.getenv("POLLY_VOICE_ID", "Matthew")
    polly_engine: str = os.getenv("POLLY_ENGINE", "neural")

    # Whisper
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")

    # Audio
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("CHANNELS", "1"))


settings = Settings()
