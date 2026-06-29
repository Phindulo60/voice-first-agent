"""
Stage 1: Automatic Speech Recognition (ASR)
Whisper-based local transcription — speech to text.
Supports English and isiZulu (via language parameter).
"""

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from rich.console import Console

from src.config import settings

console = Console()

# Load model once (lazy singleton)
_model = None


def get_model() -> WhisperModel:
    """Load Whisper model (cached after first call)."""
    global _model
    if _model is None:
        console.print(f"[dim]Loading Whisper model: {settings.whisper_model}...[/dim]")
        _model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type="int8",
        )
        console.print("[green]✓ Whisper model loaded[/green]")
    return _model


def record_audio(duration: float = None) -> np.ndarray:
    """
    Record audio from microphone.
    If duration is None, records until Enter is pressed.
    """
    sample_rate = settings.sample_rate
    channels = settings.channels

    if duration:
        console.print(f"[yellow]🎤 Recording for {duration}s...[/yellow]")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        sd.wait()
    else:
        console.print("[yellow]🎤 Recording... Press Enter to stop.[/yellow]")
        frames = []
        recording = True

        def callback(indata, frame_count, time_info, status):
            if recording:
                frames.append(indata.copy())

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=callback,
        )
        with stream:
            input()
            recording = False

        audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, channels))

    console.print("[green]✓ Recording complete[/green]")
    return audio.flatten()


def transcribe(audio: np.ndarray, language: str = None) -> str:
    """
    Transcribe audio array to text using Whisper.

    Args:
        audio: numpy array of audio samples (16kHz mono float32)
        language: Language code ('en', 'zu', etc). If None, uses config default.
    """
    model = get_model()
    lang = language or settings.asr_language

    segments, info = model.transcribe(
        audio,
        language=lang if lang != "auto" else None,  # None = auto-detect
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join(segment.text.strip() for segment in segments)
    console.print(f"[blue]📝 Transcribed ({lang}):[/blue] {text}")
    return text


# Allow running standalone for testing
if __name__ == "__main__":
    console.print("[bold]ASR Test — Speak (language from config)[/bold]\n")
    audio = record_audio()
    if len(audio) > 0:
        text = transcribe(audio)
        console.print(f"\n[bold green]Result:[/bold green] {text}")
    else:
        console.print("[red]No audio recorded.[/red]")
