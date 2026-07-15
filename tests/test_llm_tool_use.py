"""
Unit tests for the tool_use round trip in src.llm.chat(), mocking the
Bedrock client entirely — no real AWS calls, no network.
"""

import pytest

from src import llm, tools


@pytest.fixture(autouse=True)
def fresh_tool_state():
    tools.reset_state(account_balance=100.0)
    yield
    tools.reset_state(account_balance=100.0)


class FakeBedrockClient:
    """Returns a scripted sequence of converse() responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def converse(self, **kwargs):
        # Snapshot the messages list — callers mutate it in place after this
        # returns, so without a copy every captured call would alias the
        # same, later-mutated list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def _text_response(text, stop_reason="end_turn"):
    return {
        "stopReason": stop_reason,
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
    }


def _tool_use_response(tool_use_id, name, tool_input):
    return {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input}}],
            }
        },
    }


def test_chat_returns_text_when_no_tool_use(monkeypatch):
    fake = FakeBedrockClient([_text_response("Hello there!")])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    result = llm.chat("Hi", [])
    assert result == "Hello there!"
    assert len(fake.calls) == 1


def test_chat_resolves_a_single_tool_call(monkeypatch):
    fake = FakeBedrockClient([
        _tool_use_response("t1", "purchase_airtime", {"amount": 20, "provider": "MTN"}),
        _text_response("Done — you bought R20 of MTN airtime, new balance R80."),
    ])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    history = []
    result = llm.chat("Buy me R20 of MTN airtime", history)

    assert "R80" in result
    assert len(fake.calls) == 2

    # Second call must carry the tool result back to the model
    second_call_messages = fake.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    tool_result = tool_result_msg["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "t1"
    assert tool_result["content"][0]["json"]["new_balance"] == 80.0


def test_chat_mutates_history_in_place(monkeypatch):
    fake = FakeBedrockClient([_text_response("ok")])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    history = []
    llm.chat("hello", history)

    # Caller's list should now contain both the user turn and assistant reply
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_chat_invokes_on_tool_call_with_structured_event(monkeypatch):
    fake = FakeBedrockClient([
        _tool_use_response("t1", "purchase_airtime", {"amount": 20, "provider": "MTN"}),
        _text_response("Done."),
    ])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    recorded = []
    llm.chat("Buy me R20 of MTN airtime", [], on_tool_call=recorded.append)

    assert len(recorded) == 1
    event = recorded[0]
    assert event["name"] == "purchase_airtime"
    assert event["input"] == {"amount": 20, "provider": "MTN"}
    assert event["result"]["success"] is True
    assert event["latency_ms"] >= 0


def test_chat_without_on_tool_call_is_optional(monkeypatch):
    # chat() must work fine when callers do not pass on_tool_call at all.
    fake = FakeBedrockClient([
        _tool_use_response("t1", "purchase_airtime", {"amount": 20}),
        _text_response("Done."),
    ])
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    result = llm.chat("Buy me R20 of airtime", [])
    assert result == "Done."


def test_chat_gives_up_after_max_tool_rounds(monkeypatch):
    responses = [
        _tool_use_response(f"t{i}", "purchase_airtime", {"amount": 1})
        for i in range(llm.MAX_TOOL_ROUNDS)
    ]
    fake = FakeBedrockClient(responses)
    monkeypatch.setattr(llm, "get_client", lambda: fake)

    result = llm.chat("keep buying airtime forever", [])
    assert "stuck" in result.lower()
    assert len(fake.calls) == llm.MAX_TOOL_ROUNDS
