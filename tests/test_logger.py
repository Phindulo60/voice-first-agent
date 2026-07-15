"""Unit tests for session logging (research data collection)."""

import json

import pytest

from src.logger import SessionLogger


@pytest.fixture
def logger(tmp_path, monkeypatch):
    """SessionLogger writing into a temp dir instead of ~/.voice-first-agent."""
    monkeypatch.setattr("src.logger.LOG_DIR", tmp_path)
    return SessionLogger(pipeline="zulu", safety_enabled=True)


def test_new_turn_has_identity_and_mode(logger):
    turn = logger.new_turn()
    assert turn.session_id == logger.session_id
    assert turn.turn_id == f"{logger.session_id}-1"
    assert turn.pipeline == "zulu"
    assert turn.safety_enabled is True


def test_turn_ids_increment_per_session(logger):
    first = logger.new_turn()
    second = logger.new_turn()
    assert first.turn_id.endswith("-1")
    assert second.turn_id.endswith("-2")


def test_stage_timer_measures_elapsed_ms(logger):
    t0 = logger.stage_timer_start()
    elapsed = logger.stage_timer_ms(t0)
    assert elapsed >= 0


def test_finish_turn_writes_one_jsonl_line(logger):
    turn = logger.new_turn()
    turn.asr_text = "Ngiyabonga"
    logger.finish_turn(turn)

    assert logger.log_file.exists()
    lines = logger.log_file.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["asr_text"] == "Ngiyabonga"
    assert record["session_id"] == logger.session_id


def test_session_summary_counts_safety_triggers(logger):
    triggered = logger.new_turn()
    triggered.safety_triggered = True
    logger.finish_turn(triggered)

    calm = logger.new_turn()
    logger.finish_turn(calm)

    summary = logger.session_summary()
    assert summary["total_turns"] == 2
    assert summary["safety_triggered_count"] == 1
    assert summary["safety_trigger_rate"] == 0.5


def test_session_summary_empty_when_no_log_file(logger):
    assert logger.session_summary() == {}


def test_session_summary_ignores_other_sessions(logger, tmp_path, monkeypatch):
    monkeypatch.setattr("src.logger.LOG_DIR", tmp_path)
    other = SessionLogger(pipeline="english", safety_enabled=False)
    other.log_file = logger.log_file  # simulate same day, different session
    logger.finish_turn(logger.new_turn())
    other.finish_turn(other.new_turn())

    summary = logger.session_summary()
    assert summary["total_turns"] == 1
