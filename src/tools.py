"""
Tool use / function calling for the LLM stage.

First and only tool for now (see GitHub issue #4): purchase_airtime.
Deliberately narrow — proves the Bedrock Converse tool_use round trip
end-to-end against the re-scoped first use case (issue #7) before any
general MCP/tool-server integration.

Backend is mocked in-memory. Swap for a real telco API later without
touching the tool-calling plumbing in llm.py.
"""

from typing import Optional

PROVIDERS = ("MTN", "Vodacom", "Cell C", "Telkom")

# Mock account state — one user, in-memory, resets on process restart.
_state = {
    "account_balance": 100.0,
    "airtime_credit": 0.0,
}


def reset_state(account_balance: float = 100.0):
    """Reset the mock account to a known state. Used by tests and fresh sessions."""
    _state["account_balance"] = account_balance
    _state["airtime_credit"] = 0.0


def purchase_airtime(amount: float, provider: str = "MTN") -> dict:
    """
    Purchase mobile airtime. Deducts from the mock account balance and
    credits airtime. This is the "backend state changed" action from
    BENCHMARKS.md's dual success metric — new_balance is directly
    verifiable before/after.
    """
    if amount <= 0:
        return {"success": False, "new_balance": _state["account_balance"], "error": "Amount must be greater than zero"}

    if provider not in PROVIDERS:
        return {"success": False, "new_balance": _state["account_balance"], "error": f"Unknown provider '{provider}'. Choose from {', '.join(PROVIDERS)}"}

    if amount > _state["account_balance"]:
        return {"success": False, "new_balance": _state["account_balance"], "error": f"Insufficient balance: R{_state['account_balance']:.2f} available, R{amount:.2f} requested"}

    _state["account_balance"] -= amount
    _state["airtime_credit"] += amount

    return {
        "success": True,
        "new_balance": round(_state["account_balance"], 2),
        "airtime_credit": round(_state["airtime_credit"], 2),
        "error": None,
    }


# Bedrock Converse toolConfig — see:
# https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html
TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "purchase_airtime",
            "description": (
                "Purchase mobile airtime for the user, deducting from their account "
                "balance. Use this when the user asks to buy/load airtime."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Amount in ZAR (South African Rand) to purchase, e.g. 20",
                        },
                        "provider": {
                            "type": "string",
                            "description": "Mobile network provider",
                            "enum": list(PROVIDERS),
                        },
                    },
                    "required": ["amount"],
                }
            },
        }
    }
]

# Dispatch table — name (as declared in TOOL_SPECS) -> implementation.
_TOOL_FUNCTIONS = {
    "purchase_airtime": purchase_airtime,
}


def execute_tool(name: str, tool_input: dict) -> dict:
    """Run a tool by name with the args the LLM provided. Never raises."""
    fn = _TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"success": False, "error": f"Unknown tool: {name}"}

    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"success": False, "error": f"Bad arguments for {name}: {e}"}
