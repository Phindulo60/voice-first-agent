"""
Stage 2: LLM Reasoning (Bedrock Claude)
Text in → text response out via Converse API.

Supports tool use (see src/tools.py, issue #4): if the model calls a
tool, we execute it locally and feed the result back until the model
produces a final text answer.
"""

import boto3
from rich.console import Console

from src.config import settings
from src.tools import TOOL_SPECS, execute_tool

console = Console()

# Bedrock client (lazy)
_client = None

MAX_TOOL_ROUNDS = 5


def get_client():
    """Get Bedrock Runtime client."""
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )
    return _client


# System prompt — keep simple for baseline
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses concise and conversational — 
they will be spoken aloud. Aim for 1-3 sentences unless the user asks for detail. 
Do not use markdown, bullet points, or formatting — plain spoken English only.

You have access to a purchase_airtime tool. Use it whenever the user asks to buy,
load, or top up airtime. After calling it, tell the user clearly whether it
succeeded and their new balance, or what went wrong."""


def _call_bedrock(messages: list) -> dict:
    return get_client().converse(
        modelId=settings.bedrock_model_id,
        messages=messages,
        system=[{"text": SYSTEM_PROMPT}],
        toolConfig={"tools": TOOL_SPECS},
        inferenceConfig={
            "maxTokens": 200,  # Keep responses short for voice
            "temperature": 0.7,
        },
    )


def _extract_text(content_blocks: list) -> str:
    return " ".join(block["text"] for block in content_blocks if "text" in block)


def chat(user_text: str, conversation_history: list = None) -> str:
    """
    Send user text to Bedrock Claude via Converse API, resolving any
    tool_use round trips, and return the final text response.

    Args:
        user_text: Transcribed user speech
        conversation_history: List of prior messages for multi-turn.
            Mutated in place so callers keep the running history.

    Returns:
        Assistant response text
    """
    messages = conversation_history if conversation_history is not None else []
    messages.append({
        "role": "user",
        "content": [{"text": user_text}],
    })

    console.print("[dim]Thinking...[/dim]")

    for _ in range(MAX_TOOL_ROUNDS):
        response = _call_bedrock(messages)
        output_message = response["output"]["message"]
        messages.append(output_message)

        if response["stopReason"] != "tool_use":
            assistant_text = _extract_text(output_message["content"])
            console.print(f"[magenta]🤖 Response:[/magenta] {assistant_text}")
            return assistant_text

        tool_result_content = []
        for block in output_message["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            console.print(f"[cyan]🔧 Tool call:[/cyan] {tool_use['name']}({tool_use['input']})")
            result = execute_tool(tool_use["name"], tool_use["input"])
            console.print(f"[cyan]🔧 Tool result:[/cyan] {result}")
            tool_result_content.append({
                "toolResult": {
                    "toolUseId": tool_use["toolUseId"],
                    "content": [{"json": result}],
                }
            })

        messages.append({
            "role": "user",
            "content": tool_result_content,
        })

    console.print("[red]Tool loop did not resolve — too many rounds[/red]")
    return "Sorry, I got stuck trying to do that. Please try again."


# Allow running standalone for testing
if __name__ == "__main__":
    console.print("[bold]LLM Test — Type a message[/bold]\n")
    history = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            response = chat(user_input, history)
        except KeyboardInterrupt:
            break
    console.print("\n[dim]Done.[/dim]")
