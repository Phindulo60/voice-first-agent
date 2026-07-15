"""
Voice-First Agent — Main Loop

Modes:
  - English baseline: Mic → ASR → LLM → TTS
  - Zulu pipeline:    Mic → ASR → MT(zu→en) → [Safety Layer] → LLM → MT(en→zu) → TTS

Every turn is logged to ~/.voice-first-agent/sessions/ for research analysis.
See BENCHMARKS.md for the measurement framework.
"""

import sys
import time
from rich.console import Console
from rich.panel import Panel

from src.asr import record_audio, transcribe, get_model
from src.llm import chat
from src.tts import speak
from src.config import settings
from src.logger import SessionLogger

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
    logger = SessionLogger(pipeline="english", safety_enabled=False)

    while True:
        try:
            input("\n[Press Enter to start speaking...]")
            turn = logger.new_turn()
            turn_start = time.time()

            audio = record_audio()
            if len(audio) < 1600:
                console.print("[dim]No audio detected, try again.[/dim]")
                continue

            t0 = logger.stage_timer_start()
            text = transcribe(audio, language="en")
            turn.asr_engine = "whisper"
            turn.asr_text = text
            turn.asr_latency_ms = logger.stage_timer_ms(t0)

            if not text.strip():
                console.print("[dim]Couldn't understand, try again.[/dim]")
                continue

            if text.strip().lower() in ("goodbye", "exit", "quit", "stop"):
                speak("Goodbye!")
                logger.finish_turn(turn, turn_start)
                break

            t0 = logger.stage_timer_start()
            response = chat(text, conversation_history)
            turn.llm_input_text = text
            turn.llm_response_text = response
            turn.llm_latency_ms = logger.stage_timer_ms(t0)

            t0 = logger.stage_timer_start()
            speak(response)
            turn.tts_engine = settings.tts_engine
            turn.tts_text = response
            turn.tts_latency_ms = logger.stage_timer_ms(t0)

            logger.finish_turn(turn, turn_start)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if 'turn' in locals():
                turn.error = str(e)
                logger.finish_turn(turn, turn_start)
            continue

    _print_summary(logger)


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
    logger = SessionLogger(pipeline="zulu", safety_enabled=settings.safety_enabled)

    while True:
        try:
            input("\n[Press Enter to start speaking...]")
            turn = logger.new_turn()
            turn_start = time.time()

            audio = record_audio()
            if len(audio) < 1600:
                console.print("[dim]No audio detected, try again.[/dim]")
                continue

            # Stage 1: ASR
            t0 = logger.stage_timer_start()
            zulu_text = transcribe(audio)
            turn.asr_engine = "mms"
            turn.asr_text = zulu_text
            turn.asr_latency_ms = logger.stage_timer_ms(t0)

            if not zulu_text.strip():
                console.print("[dim]Couldn't understand, try again.[/dim]")
                continue

            if zulu_text.strip().lower() in ("sala kahle", "exit", "quit", "stop"):
                speak("Sala kahle!")
                logger.finish_turn(turn, turn_start)
                break

            # Stage 2: MT zu->en
            t0 = logger.stage_timer_start()
            english_text = zulu_to_english(zulu_text)
            turn.mt_forward_text = english_text
            turn.mt_forward_latency_ms = logger.stage_timer_ms(t0)

            # Stage 2.5: Safety Layer
            if settings.safety_enabled:
                t0 = logger.stage_timer_start()
                zulu_back = english_to_zulu(english_text)
                safety = check_confidence(zulu_text, zulu_back)
                console.print(f"  {format_confidence(safety)}")

                turn.safety_confidence = safety.confidence.value
                turn.safety_similarity = safety.similarity
                turn.safety_back_translation = zulu_back

                if safety.confidence == Confidence.LOW:
                    turn.safety_triggered = True
                    console.print("[yellow]⚠️  Low confidence — asking user to confirm[/yellow]")
                    confirm_zulu = f"Ngizwe uthi: {zulu_back}. Kulungile?"
                    console.print(f"[cyan]🗣️  Confirm prompt:[/cyan] {confirm_zulu}")
                    speak(confirm_zulu)

                    input("\n[Press Enter to respond...]")
                    confirm_audio = record_audio()
                    if len(confirm_audio) < 1600:
                        console.print("[dim]No response — skipping this turn.[/dim]")
                        turn.safety_user_confirmed = False
                        logger.finish_turn(turn, turn_start)
                        continue

                    confirm_text = transcribe(confirm_audio)
                    confirmed = is_affirmative(confirm_text)
                    turn.safety_user_confirmed = confirmed

                    if not confirmed:
                        console.print("[dim]User declined — asking again.[/dim]")
                        speak("Kulungile, phinda futhi.")
                        logger.finish_turn(turn, turn_start)
                        continue

                    console.print("[green]✓ Confirmed, proceeding.[/green]")

                elif safety.confidence == Confidence.MED:
                    console.print("[yellow]⚠️  Medium confidence — proceeding with implicit confirm[/yellow]")

            # Stage 3: LLM
            t0 = logger.stage_timer_start()
            english_response = chat(english_text, conversation_history)
            turn.llm_input_text = english_text
            turn.llm_response_text = english_response
            turn.llm_latency_ms = logger.stage_timer_ms(t0)

            # Stage 4: MT en->zu
            t0 = logger.stage_timer_start()
            zulu_response = english_to_zulu(english_response)
            turn.mt_reverse_text = zulu_response
            turn.mt_reverse_latency_ms = logger.stage_timer_ms(t0)

            # Stage 5: TTS
            console.print(f"[bold green]🗣️  isiZulu:[/bold green] {zulu_response}")
            t0 = logger.stage_timer_start()
            speak(zulu_response)
            turn.tts_engine = settings.tts_engine
            turn.tts_text = zulu_response
            turn.tts_latency_ms = logger.stage_timer_ms(t0)

            logger.finish_turn(turn, turn_start)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Sala kahle![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if 'turn' in locals():
                turn.error = str(e)
                logger.finish_turn(turn, turn_start)
            continue

    _print_summary(logger)


def _print_summary(logger: SessionLogger):
    """Print a quick session summary at the end."""
    summary = logger.session_summary()
    if not summary:
        return

    console.print("\n[bold]Session Summary[/bold]")
    console.print(f"  Session ID: {summary['session_id']}")
    console.print(f"  Turns: {summary['total_turns']}")
    console.print(f"  Safety triggered: {summary['safety_triggered_count']} ({summary['safety_trigger_rate']*100:.0f}%)")
    console.print(f"  Avg latency: {summary['avg_total_latency_ms']:.0f}ms")
    console.print(f"  Log file: {logger.log_file}")


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
                "  SAFETY_LOW_THRESHOLD=0.50   Low confidence cutoff\n\n"
                "Session logs written to ~/.voice-first-agent/sessions/\n"
                "Analyze with: python -m src.logger [path/to/log.jsonl]"
            )
            return

    if mode == "english":
        run_english()
    else:
        run_zulu()

    console.print("\n[bold]Session ended.[/bold]")


if __name__ == "__main__":
    main()
