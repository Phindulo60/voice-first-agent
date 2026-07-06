"""
Voice-First Agent — Main Loop

Modes:
  - English baseline: Mic → ASR → LLM → TTS
  - Zulu pipeline:    Mic → ASR → MT(zu→en) → [Safety Layer] → LLM → MT(en→zu) → TTS
"""

import sys
from rich.console import Console
from rich.panel import Panel

from src.asr import record_audio, transcribe, get_model
from src.llm import chat
from src.tts import speak
from src.config import settings

console = Console()


def run_english():
    """English-only pipeline: ASR → LLM → TTS"""
    console.print(
        Panel(
            "[bold]Voice-First Agent — English Baseline[/bold]\n\n"
            "[dim]Pipeline:[/dim] Mic → Whisper ASR → Bedrock Claude → F5-TTS → Speaker",
            title="🎤 English Mode",
            border_style="blue",
        )
    )

    get_model()
    conversation_history = []

    while True:
        try:
            input("\n[Press Enter to start speaking...]")
            audio = record_audio()

            if len(audio) < 1600:
                console.print("[dim]No audio detected, try again.[/dim]")
                continue

            text = transcribe(audio, language="en")
            if not text.strip():
                console.print("[dim]Couldn't understand, try again.[/dim]")
                continue

            if text.strip().lower() in ("goodbye", "exit", "quit", "stop"):
                speak("Goodbye!")
                break

            response = chat(text, conversation_history)
            speak(response)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue


def run_zulu():
    """Zulu pipeline: ASR → MT → [Safety] → LLM → MT → TTS"""
    from src.mt import zulu_to_english, english_to_zulu
    from src.safety import check_confidence, format_confidence, Confidence, is_affirmative

    safety_status = "ON" if settings.safety_enabled else "OFF"

    console.print(
        Panel(
            f"[bold]Voice-First Agent — isiZulu Pipeline[/bold]\n\n"
            f"[dim]Pipeline:[/dim] Mic → MMS ASR → NLLB (zu→en) → [Safety: {safety_status}] → Claude → NLLB (en→zu) → F5-TTS\n\n"
            f"[dim]Safety thresholds:[/dim] high ≥ {settings.safety_high_threshold}, low ≥ {settings.safety_low_threshold}\n\n"
            "Say [bold]'sala kahle'[/bold] to quit.",
            title="🎤 isiZulu Mode",
            border_style="green",
        )
    )

    get_model()
    conversation_history = []

    while True:
        try:
            input("\n[Press Enter to start speaking...]")
            audio = record_audio()

            if len(audio) < 1600:
                console.print("[dim]No audio detected, try again.[/dim]")
                continue

            # Stage 1: ASR — Zulu speech to text
            zulu_text = transcribe(audio)
            if not zulu_text.strip():
                console.print("[dim]Couldn't understand, try again.[/dim]")
                continue

            if zulu_text.strip().lower() in ("sala kahle", "exit", "quit", "stop"):
                speak("Sala kahle!")
                break

            # Stage 2: MT — Zulu text → English
            english_text = zulu_to_english(zulu_text)

            # Stage 2.5: Safety Layer — back-translate and check confidence
            if settings.safety_enabled:
                zulu_back = english_to_zulu(english_text)
                safety = check_confidence(zulu_text, zulu_back)
                console.print(f"  {format_confidence(safety)}")

                if safety.confidence == Confidence.LOW:
                    # Ask the user to confirm before proceeding
                    console.print("[yellow]⚠️  Low confidence — asking user to confirm[/yellow]")
                    confirm_zulu = f"Ngizwe uthi: {zulu_back}. Kulungile?"
                    console.print(f"[cyan]🗣️  Confirm prompt:[/cyan] {confirm_zulu}")
                    speak(confirm_zulu)

                    # Get confirmation
                    input("\n[Press Enter to respond...]")
                    confirm_audio = record_audio()
                    if len(confirm_audio) < 1600:
                        console.print("[dim]No response — skipping this turn.[/dim]")
                        continue

                    confirm_text = transcribe(confirm_audio)
                    if not is_affirmative(confirm_text):
                        console.print("[dim]User declined — asking again.[/dim]")
                        speak("Kulungile, phinda futhi.")
                        continue

                    console.print("[green]✓ Confirmed, proceeding.[/green]")

                elif safety.confidence == Confidence.MED:
                    # Implicit confirm — proceed but note the uncertainty
                    console.print("[yellow]⚠️  Medium confidence — proceeding with implicit confirm[/yellow]")

            # Stage 3: LLM — English reasoning
            english_response = chat(english_text, conversation_history)

            # Stage 4: MT — English response → Zulu
            zulu_response = english_to_zulu(english_response)

            # Stage 5: TTS — Speak the Zulu response
            console.print(f"[bold green]🗣️  isiZulu:[/bold green] {zulu_response}")
            speak(zulu_response)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Sala kahle![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue


def main():
    """Entry point — choose mode."""
    mode = "zulu"

    if len(sys.argv) > 1:
        if sys.argv[1] in ("--english", "-e", "en"):
            mode = "english"
        elif sys.argv[1] in ("--zulu", "-z", "zu"):
            mode = "zulu"
        elif sys.argv[1] in ("--help", "-h"):
            console.print(
                "[bold]Usage:[/bold] python -m src.main [--english | --zulu]\n\n"
                "  --english, -e    English-only pipeline\n"
                "  --zulu, -z       isiZulu pipeline [default]\n\n"
                "Environment variables:\n"
                "  SAFETY_ENABLED=true|false   Toggle the safety layer\n"
                "  SAFETY_HIGH_THRESHOLD=0.75  High confidence cutoff\n"
                "  SAFETY_LOW_THRESHOLD=0.50   Low confidence cutoff"
            )
            return

    if mode == "english":
        run_english()
    else:
        run_zulu()

    console.print("\n[bold]Session ended.[/bold]")


if __name__ == "__main__":
    main()
