"""
Stage 1: Automatic Speech Recognition (ASR)
Whisper-based local transcription — English voice → text.
"""

import io
import numpy as np
import sounddevice as sd
import soundfile as sf
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
            device="cpu",  # Use "cuda" if you have GPU
            compute_type="int8",
        )
        console.print("[green]✓ Whisper model loaded[/green]")
    return _model


def record_audio(duration: float = None) -> np.ndarray:
    """
    Record audio from microphone.
    
    If duration is None, records until Enter is pressed.
    Returns numpy array of audio samples.
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
        # Record until Enter is pressed
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
            input()  # Wait for Enter
            recording = False

        audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, channels))

    console.print("[green]✓ Recording complete[/green]")
    return audio.flatten()


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio array to text using Whisper."""
    model = get_model()

    # Whisper expects 16kHz mono float32
    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,  # Filter out silence
    )

    text = " ".join(segment.text.strip() for segment in segments)
    console.print(f"[blue]📝 Transcribed:[/blue] {text}")
    return text


# Allow running standalone for testing
if __name__ == "__main__":
    from src.config import settings  # noqa

    console.print("[bold]ASR Test — Speak in English[/bold]\n")
    audio = record_audio()
    if len(audio) > 0:
        text = transcribe(audio)
        console.print(f"\n[bold green]Result:[/bold green] {text}")
    else:
        console.print("[red]No audio recorded.[/red]")
