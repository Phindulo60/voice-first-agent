"""
Voice-First Agent — Main Loop
English voice-in → Bedrock Claude → English voice-out

The cascade:
  [Mic] → [Whisper ASR] → [Bedrock Claude] → [Amazon Polly TTS] → [Speaker]
"""

from rich.console import Console
from rich.panel import Panel

from src.asr import record_audio, transcribe, get_model
from src.llm import chat
from src.tts import speak

console = Console()


def main():
    """Main conversation loop."""
    console.print(
        Panel(
            "[bold]Voice-First Agent — English Baseline[/bold]\n\n"
            "[dim]Architecture:[/dim] Mic → Whisper ASR → Bedrock Claude → Polly TTS → Speaker\n\n"
            "Press [bold]Enter[/bold] to start recording, [bold]Enter[/bold] again to stop.\n"
            "Say [bold]'goodbye'[/bold] or [bold]'exit'[/bold] to quit.",
            title="🎤 Voice Agent",
            border_style="blue",
        )
    )

    # Pre-load Whisper model
    get_model()

    # Conversation history for multi-turn
    conversation_history = []

    while True:
        try:
            # Wait for user to be ready
            input("\n[Press Enter to start speaking...]")

            # Stage 1: Record + Transcribe
            audio = record_audio()
            if len(audio) < 1600:  # Less than 0.1s of audio
                console.print("[dim]No audio detected, try again.[/dim]")
                continue

            text = transcribe(audio)
            if not text.strip():
                console.print("[dim]Couldn't understand, try again.[/dim]")
                continue

            # Check for exit
            if text.strip().lower() in ("goodbye", "exit", "quit", "stop"):
                speak("Goodbye!")
                break

            # Stage 2: LLM reasoning
            response = chat(text, conversation_history)

            # Stage 3: Text-to-Speech
            speak(response)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    console.print("\n[bold]Session ended.[/bold]")


if __name__ == "__main__":
    main()
