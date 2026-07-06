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

    # TTS
    tts_engine: str = os.getenv("TTS_ENGINE", "mms")  # 'mms' (native Zulu) or 'f5' (voice cloning)
    mms_tts_model: str = os.getenv("MMS_TTS_MODEL", "facebook/mms-tts-zul")
    tts_device: str = os.getenv("TTS_DEVICE", "cpu")
    tts_ref_audio: str = os.getenv("TTS_REF_AUDIO", "")
    tts_ref_text: str = os.getenv("TTS_REF_TEXT", "")

    # NLLB Translation
    nllb_model: str = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
    mt_device: str = os.getenv("MT_DEVICE", "cpu")

    # ASR
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    mms_model: str = os.getenv("MMS_MODEL", "facebook/mms-1b-all")
    asr_language: str = os.getenv("ASR_LANGUAGE", "zu")

    # Safety Layer
    safety_enabled: bool = os.getenv("SAFETY_ENABLED", "true").lower() == "true"
    safety_high_threshold: float = float(os.getenv("SAFETY_HIGH_THRESHOLD", "0.75"))
    safety_low_threshold: float = float(os.getenv("SAFETY_LOW_THRESHOLD", "0.50"))

    # Audio
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("CHANNELS", "1"))


settings = Settings()
