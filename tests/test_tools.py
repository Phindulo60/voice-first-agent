"""Unit tests for the purchase_airtime tool (issue #4 / #7)."""

import pytest

from src import tools


@pytest.fixture(autouse=True)
def fresh_state():
    """Every test starts from a known mock account balance."""
    tools.reset_state(account_balance=100.0)
    yield
    tools.reset_state(account_balance=100.0)


def test_successful_purchase_deducts_balance():
    result = tools.purchase_airtime(20, "MTN")
    assert result["success"] is True
    assert result["new_balance"] == 80.0
    assert result["airtime_credit"] == 20.0
    assert result["error"] is None


def test_default_provider_is_mtn():
    result = tools.purchase_airtime(10)
    assert result["success"] is True


def test_rejects_non_positive_amount():
    result = tools.purchase_airtime(0, "MTN")
    assert result["success"] is False
    assert "greater than zero" in result["error"]


def test_rejects_unknown_provider():
    result = tools.purchase_airtime(10, "Nonexistent Telco")
    assert result["success"] is False
    assert "Unknown provider" in result["error"]


def test_rejects_insufficient_balance():
    result = tools.purchase_airtime(1000, "Vodacom")
    assert result["success"] is False
    assert "Insufficient balance" in result["error"]
    assert result["new_balance"] == 100.0  # unchanged


def test_balance_persists_across_purchases():
    tools.purchase_airtime(30, "MTN")
    second = tools.purchase_airtime(20, "Vodacom")
    assert second["new_balance"] == 50.0
    assert second["airtime_credit"] == 50.0


def test_execute_tool_dispatches_by_name():
    result = tools.execute_tool("purchase_airtime", {"amount": 15, "provider": "Telkom"})
    assert result["success"] is True
    assert result["new_balance"] == 85.0


def test_execute_tool_unknown_name():
    result = tools.execute_tool("send_rocket_to_moon", {})
    assert result["success"] is False
    assert "Unknown tool" in result["error"]


def test_execute_tool_bad_arguments():
    result = tools.execute_tool("purchase_airtime", {"unexpected_field": 1})
    assert result["success"] is False
    assert "Bad arguments" in result["error"]


def test_tool_specs_shape_matches_bedrock_converse_schema():
    assert len(tools.TOOL_SPECS) == 1
    spec = tools.TOOL_SPECS[0]["toolSpec"]
    assert spec["name"] == "purchase_airtime"
    schema = spec["inputSchema"]["json"]
    assert schema["required"] == ["amount"]
    assert "amount" in schema["properties"]
