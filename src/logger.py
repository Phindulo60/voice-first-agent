"""
Session Logging for Research Data Collection

Every voice turn gets logged as structured JSONL — this is the pipeline
as a research instrument. Without this, none of BENCHMARKS.md is measurable.

Storage: ~/.voice-first-agent/sessions/YYYY-MM-DD.jsonl
One JSON object per line, one line per conversational turn.
"""

import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional

LOG_DIR = Path.home() / ".voice-first-agent" / "sessions"


@dataclass
class TurnLog:
    """Everything captured for a single conversational turn."""

    # Identity
    session_id: str
    turn_id: str
    timestamp: str

    # Mode
    pipeline: str  # 'english' or 'zulu'
    safety_enabled: bool

    # Stage 1: ASR
    asr_engine: str = ""
    asr_text: str = ""
    asr_latency_ms: float = 0.0

    # Stage 2: MT (zu->en), only for zulu pipeline
    mt_forward_text: str = ""
    mt_forward_latency_ms: float = 0.0

    # Stage 2.5: Safety layer
    safety_triggered: bool = False
    safety_confidence: str = ""       # 'high', 'med', 'low', or '' if disabled
    safety_similarity: float = 0.0
    safety_back_translation: str = ""
    safety_user_confirmed: Optional[bool] = None  # None if not triggered

    # Stage 3: LLM
    llm_input_text: str = ""
    llm_response_text: str = ""
    llm_latency_ms: float = 0.0

    # Stage 3.5: Tool use (see issue #4) — one entry per tool call this turn:
    # {"name": str, "input": dict, "result": dict, "latency_ms": float}
    tool_calls: list = field(default_factory=list)

    # Stage 4: MT (en->zu), only for zulu pipeline
    mt_reverse_text: str = ""
    mt_reverse_latency_ms: float = 0.0

    # Stage 5: TTS
    tts_engine: str = ""
    tts_text: str = ""
    tts_latency_ms: float = 0.0

    # End-to-end
    total_latency_ms: float = 0.0

    # Outcome (filled in manually or by task-specific logic later)
    task_completed: Optional[bool] = None
    notes: str = ""

    # Errors
    error: str = ""


class SessionLogger:
    """
    Tracks one session (a run of `python -m src.main`) and logs
    each turn to a daily JSONL file.
    """

    def __init__(self, pipeline: str, safety_enabled: bool):
        self.session_id = str(uuid.uuid4())[:8]
        self.pipeline = pipeline
        self.safety_enabled = safety_enabled
        self._turn_counter = 0
        self._stage_start = None

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.log_file = LOG_DIR / f"{today}.jsonl"

    def new_turn(self) -> TurnLog:
        """Start tracking a new turn."""
        self._turn_counter += 1
        self._turn_start = time.time()
        return TurnLog(
            session_id=self.session_id,
            turn_id=f"{self.session_id}-{self._turn_counter}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline=self.pipeline,
            safety_enabled=self.safety_enabled,
        )

    def stage_timer_start(self) -> float:
        """Call before a stage begins."""
        return time.time()

    def stage_timer_ms(self, start: float) -> float:
        """Call after a stage ends, with the value from stage_timer_start()."""
        return round((time.time() - start) * 1000, 1)

    def finish_turn(self, turn: TurnLog, turn_start: float = None):
        """Write the completed turn to disk."""
        if turn_start:
            turn.total_latency_ms = round((time.time() - turn_start) * 1000, 1)
        elif hasattr(self, "_turn_start"):
            turn.total_latency_ms = round((time.time() - self._turn_start) * 1000, 1)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")

    def session_summary(self) -> dict:
        """Read back this session's turns from the log file and summarize."""
        if not self.log_file.exists():
            return {}

        turns = []
        with open(self.log_file) as f:
            for line in f:
                data = json.loads(line)
                if data.get("session_id") == self.session_id:
                    turns.append(data)

        if not turns:
            return {}

        safety_fires = sum(1 for t in turns if t.get("safety_triggered"))
        completed = sum(1 for t in turns if t.get("task_completed") is True)

        all_tool_calls = [call for t in turns for call in t.get("tool_calls", [])]
        tool_calls_succeeded = sum(1 for call in all_tool_calls if call.get("result", {}).get("success") is True)

        return {
            "session_id": self.session_id,
            "pipeline": self.pipeline,
            "total_turns": len(turns),
            "safety_triggered_count": safety_fires,
            "safety_trigger_rate": round(safety_fires / len(turns), 2) if turns else 0,
            "tasks_completed": completed,
            "tool_calls_total": len(all_tool_calls),
            "tool_calls_succeeded": tool_calls_succeeded,
            "tool_call_success_rate": round(tool_calls_succeeded / len(all_tool_calls), 2) if all_tool_calls else None,
            "avg_total_latency_ms": round(
                sum(t.get("total_latency_ms", 0) for t in turns) / len(turns), 1
            ),
        }


# Quick analysis helper — run standalone to summarize a day's logs
if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if len(sys.argv) > 1:
        target_file = Path(sys.argv[1])
    else:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_file = LOG_DIR / f"{today}.jsonl"

    if not target_file.exists():
        console.print(f"[red]No log file found: {target_file}[/red]")
        sys.exit(1)

    turns = []
    with open(target_file) as f:
        for line in f:
            turns.append(json.loads(line))

    console.print(f"[bold]Log file:[/bold] {target_file}")
    console.print(f"[bold]Total turns:[/bold] {len(turns)}\n")

    table = Table(title="Turn Summary")
    table.add_column("Turn")
    table.add_column("Pipeline")
    table.add_column("ASR")
    table.add_column("Safety")
    table.add_column("Latency (ms)")

    for t in turns:
        safety_str = (
            f"{t.get('safety_confidence', '-')} ({t.get('safety_similarity', 0):.2f})"
            if t.get("safety_enabled")
            else "off"
        )
        table.add_row(
            t.get("turn_id", ""),
            t.get("pipeline", ""),
            (t.get("asr_text", "") or "")[:30],
            safety_str,
            str(t.get("total_latency_ms", "-")),
        )

    console.print(table)
