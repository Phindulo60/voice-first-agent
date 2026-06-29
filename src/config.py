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

    # F5-TTS
    tts_device: str = os.getenv("TTS_DEVICE", "cpu")
    tts_ref_audio: str = os.getenv("TTS_REF_AUDIO", "")
    tts_ref_text: str = os.getenv("TTS_REF_TEXT", "")

    # NLLB Translation
    nllb_model: str = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
    mt_device: str = os.getenv("MT_DEVICE", "cpu")

    # ASR
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    mms_model: str = os.getenv("MMS_MODEL", "facebook/mms-1b-all")
    asr_language: str = os.getenv("ASR_LANGUAGE", "zu")  # 'zu' for Zulu (MMS), 'en' for English (Whisper)

    # Audio
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("CHANNELS", "1"))


settings = Settings()
