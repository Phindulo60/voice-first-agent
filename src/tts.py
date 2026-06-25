"""
Stage 3: Text-to-Speech (F5-TTS)
Text → English speech audio → playback.

F5-TTS: Flow-matching based TTS, runs locally.
Uses a reference audio for voice cloning (zero-shot).
Falls back to built-in example voice if no reference provided.
"""

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console

from src.config import settings

console = Console()

# F5-TTS model (lazy singleton)
_model = None


def get_model():
    """Load F5-TTS model (cached after first call)."""
    global _model
    if _model is None:
        console.print("[dim]Loading F5-TTS model...[/dim]")
        from f5_tts.api import F5TTS
        _model = F5TTS(model_type="F5-TTS", ckpt_file="", device=settings.tts_device)
        console.print("[green]✓ F5-TTS model loaded[/green]")
    return _model


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """
    Convert text to speech using F5-TTS.
    
    Returns:
        Tuple of (audio numpy array, sample rate)
    """
    model = get_model()

    # Use reference audio for voice style (or F5-TTS built-in example)
    ref_file = settings.tts_ref_audio
    ref_text = settings.tts_ref_text

    # Generate speech
    wav, sr, _ = model.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=text,
        seed=None,  # Random seed for variation
    )

    return wav, sr


def speak(text: str):
    """Synthesize text and play it through speakers."""
    console.print("[dim]Speaking...[/dim]")

    wav, sr = synthesize(text)

    # Play audio
    sd.play(wav, samplerate=sr)
    sd.wait()  # Block until playback finishes

    console.print("[green]✓ Done speaking[/green]")


def save_audio(text: str, output_path: str):
    """Synthesize and save to file instead of playing."""
    wav, sr = synthesize(text)
    sf.write(output_path, wav, sr)
    console.print(f"[green]✓ Saved to {output_path}[/green]")


# Allow running standalone for testing
if __name__ == "__main__":
    console.print("[bold]TTS Test (F5-TTS) — Type text to speak[/bold]\n")
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
