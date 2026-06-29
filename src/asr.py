"""
Stage 1: Automatic Speech Recognition (ASR)

Two engines:
  - Whisper: for English (excellent)
  - MMS (Meta Massively Multilingual Speech): for isiZulu

Default Zulu model: asr-africa/mms-1B_all_NCHLT_ZULU_50hr_v1
(fine-tuned on 50hrs of Zulu NCHLT speech corpus, ~20% WER)
"""

import numpy as np
import sounddevice as sd
from rich.console import Console

from src.config import settings

console = Console()

# Models (lazy singletons)
_whisper_model = None
_mms_model = None
_mms_processor = None


def get_whisper_model():
    """Load Whisper model for English ASR."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        console.print(f"[dim]Loading Whisper model: {settings.whisper_model}...[/dim]")
        _whisper_model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type="int8",
        )
        console.print("[green]✓ Whisper model loaded[/green]")
    return _whisper_model


def get_mms_model():
    """Load MMS model for Zulu ASR."""
    global _mms_model, _mms_processor
    if _mms_model is None:
        import torch
        from transformers import Wav2Vec2ForCTC, AutoProcessor

        model_id = settings.mms_model
        console.print(f"[dim]Loading MMS model: {model_id}...[/dim]")

        _mms_processor = AutoProcessor.from_pretrained(model_id)
        _mms_model = Wav2Vec2ForCTC.from_pretrained(model_id)

        # Only load adapter if using the generic mms-1b-all model
        # Fine-tuned models (like NCHLT_ZULU) are already Zulu-specific
        if "mms-1b-all" in model_id and "NCHLT" not in model_id:
            _mms_processor.tokenizer.set_target_lang("zul")
            _mms_model.load_adapter("zul")
            console.print("[green]✓ MMS model loaded (Zulu adapter active)[/green]")
        else:
            console.print("[green]✓ MMS Zulu model loaded (fine-tuned)[/green]")

    return _mms_model, _mms_processor


def get_model():
    """Pre-load the appropriate model based on config."""
    if settings.asr_language == "en":
        get_whisper_model()
    else:
        get_mms_model()


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
    Transcribe audio to text.
    Uses Whisper for English, MMS for Zulu.
    """
    import torch

    lang = language or settings.asr_language

    if lang == "en":
        model = get_whisper_model()
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments)

    else:
        # MMS for Zulu
        model, processor = get_mms_model()

        inputs = processor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        text = processor.decode(ids)

    console.print(f"[blue]📝 Transcribed ({lang}):[/blue] {text}")
    return text


# Allow running standalone for testing
if __name__ == "__main__":
    console.print("[bold]ASR Test[/bold]\n")
    console.print(f"[dim]Language: {settings.asr_language}[/dim]")
    console.print(f"[dim]Model: {settings.mms_model if settings.asr_language != 'en' else settings.whisper_model}[/dim]\n")

    get_model()
    audio = record_audio()
    if len(audio) > 0:
        text = transcribe(audio)
        console.print(f"\n[bold green]Result:[/bold green] {text}")
    else:
        console.print("[red]No audio recorded.[/red]")
