"""
Stage 3: Text-to-Speech

Two engines:
  - MMS TTS (facebook/mms-tts-zul): native Zulu voice via VITS
  - F5-TTS: multilingual with voice cloning (fallback)
"""

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console

from src.config import settings

console = Console()

# Model singletons
_mms_tts_model = None
_mms_tts_tokenizer = None
_f5_model = None


def get_mms_tts():
    """Load MMS TTS model for native Zulu speech."""
    global _mms_tts_model, _mms_tts_tokenizer
    if _mms_tts_model is None:
        from transformers import VitsModel, AutoTokenizer

        model_id = settings.mms_tts_model
        token = os.getenv("HF_TOKEN")

        console.print(f"[dim]Loading MMS TTS model: {model_id}...[/dim]")

        kwargs = {"token": token} if token else {}
        _mms_tts_tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        _mms_tts_model = VitsModel.from_pretrained(model_id, **kwargs)
        _mms_tts_model.eval()

        console.print("[green]✓ MMS TTS model loaded (native Zulu voice)[/green]")

    return _mms_tts_model, _mms_tts_tokenizer


def get_f5_model():
    """Load F5-TTS (fallback for English or custom voices)."""
    global _f5_model
    if _f5_model is None:
        console.print("[dim]Loading F5-TTS model...[/dim]")
        from f5_tts.api import F5TTS
        _f5_model = F5TTS(model="F5TTS_v1_Base", device=settings.tts_device)
        console.print("[green]✓ F5-TTS model loaded[/green]")
    return _f5_model


def synthesize_mms(text: str) -> tuple[np.ndarray, int]:
    """Synthesize speech using MMS TTS (native Zulu)."""
    import torch

    model, tokenizer = get_mms_tts()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    audio = output.squeeze().cpu().numpy()
    sample_rate = model.config.sampling_rate
    return audio, sample_rate


def synthesize_f5(text: str) -> tuple[np.ndarray, int]:
    """Synthesize speech using F5-TTS (voice cloning)."""
    from importlib.resources import files

    model = get_f5_model()
    ref_file = settings.tts_ref_audio
    ref_text = settings.tts_ref_text

    if not ref_file:
        ref_file = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        ref_text = "Some call me nature, others call me mother nature."

    wav, sr, _ = model.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=text,
        show_info=lambda x: None,
        seed=None,
    )
    return wav, sr


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Route to the configured TTS engine."""
    if settings.tts_engine == "f5":
        return synthesize_f5(text)
    else:
        return synthesize_mms(text)


def speak(text: str):
    """Synthesize and play through speakers."""
    console.print(f"[dim]Speaking ({settings.tts_engine})...[/dim]")
    wav, sr = synthesize(text)
    sd.play(wav, samplerate=sr)
    sd.wait()
    console.print("[green]✓ Done speaking[/green]")


def save_audio(text: str, output_path: str):
    """Synthesize and save to file."""
    wav, sr = synthesize(text)
    sf.write(output_path, wav, sr)
    console.print(f"[green]✓ Saved to {output_path}[/green]")


if __name__ == "__main__":
    console.print("[bold]TTS Test[/bold]\n")
    console.print(f"[dim]Engine: {settings.tts_engine}[/dim]\n")

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
