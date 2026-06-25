"""
Stage 3: Text-to-Speech (Amazon Polly)
Text → English speech audio → playback.
"""

import io
import boto3
import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console

from src.config import settings

console = Console()

# Polly client (lazy)
_client = None


def get_client():
    """Get Amazon Polly client."""
    global _client
    if _client is None:
        _client = boto3.client(
            "polly",
            region_name=settings.aws_region,
        )
    return _client


def synthesize(text: str) -> np.ndarray:
    """
    Convert text to speech using Amazon Polly.
    Returns audio as numpy array.
    """
    client = get_client()

    response = client.synthesize_speech(
        Text=text,
        OutputFormat="pcm",
        SampleRate=str(settings.sample_rate),
        VoiceId=settings.polly_voice_id,
        Engine=settings.polly_engine,
    )

    # Read PCM audio stream
    audio_stream = response["AudioStream"].read()

    # Convert PCM bytes to numpy array (16-bit signed int → float32)
    audio = np.frombuffer(audio_stream, dtype=np.int16).astype(np.float32) / 32768.0

    return audio


def speak(text: str):
    """Synthesize text and play it through speakers."""
    console.print(f"[dim]Speaking...[/dim]")

    audio = synthesize(text)

    # Play audio
    sd.play(audio, samplerate=settings.sample_rate)
    sd.wait()  # Block until playback finishes

    console.print("[green]✓ Done speaking[/green]")


# Allow running standalone for testing
if __name__ == "__main__":
    console.print("[bold]TTS Test — Type text to speak[/bold]\n")
    while True:
        try:
            text = input("\nText: ").strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                break
            speak(text)
        except KeyboardInterrupt:
            break
    console.print("\n[dim]Done.[/dim]")
