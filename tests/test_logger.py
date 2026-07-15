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
    assert turn.tool_calls == []


def test_tool_calls_default_list_is_independent_per_turn(logger):
    """dataclass default_factory should give each turn its own list, not a shared one."""
    first = logger.new_turn()
    second = logger.new_turn()
    first.tool_calls.append({"name": "purchase_airtime"})
    assert second.tool_calls == []


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


def test_session_summary_with_no_tool_calls(logger):
    logger.finish_turn(logger.new_turn())

    summary = logger.session_summary()
    assert summary["tool_calls_total"] == 0
    assert summary["tool_calls_succeeded"] == 0
    assert summary["tool_call_success_rate"] is None


def test_session_summary_counts_tool_call_success_rate(logger):
    successful = logger.new_turn()
    successful.tool_calls.append({
        "name": "purchase_airtime",
        "input": {"amount": 20},
        "result": {"success": True, "new_balance": 80.0},
        "latency_ms": 12.3,
    })
    logger.finish_turn(successful)

    failed = logger.new_turn()
    failed.tool_calls.append({
        "name": "purchase_airtime",
        "input": {"amount": 500},
        "result": {"success": False, "error": "Insufficient balance"},
        "latency_ms": 8.1,
    })
    logger.finish_turn(failed)

    summary = logger.session_summary()
    assert summary["tool_calls_total"] == 2
    assert summary["tool_calls_succeeded"] == 1
    assert summary["tool_call_success_rate"] == 0.5


def test_session_summary_counts_multiple_tool_calls_in_one_turn(logger):
    turn = logger.new_turn()
    turn.tool_calls.append({"name": "purchase_airtime", "result": {"success": True}})
    turn.tool_calls.append({"name": "purchase_airtime", "result": {"success": True}})
    logger.finish_turn(turn)

    summary = logger.session_summary()
    assert summary["tool_calls_total"] == 2
    assert summary["tool_call_success_rate"] == 1.0
