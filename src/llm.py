"""
Stage 2: LLM Reasoning (Bedrock Claude)
Text in → text response out via Converse API.
"""

import boto3
from rich.console import Console

from src.config import settings

console = Console()

# Bedrock client (lazy)
_client = None


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
Do not use markdown, bullet points, or formatting — plain spoken English only."""


def chat(user_text: str, conversation_history: list = None) -> str:
    """
    Send user text to Bedrock Claude via Converse API.
    Returns the assistant's text response.
    
    Args:
        user_text: Transcribed user speech
        conversation_history: List of prior messages for multi-turn
    
    Returns:
        Assistant response text
    """
    client = get_client()

    # Build messages
    messages = conversation_history or []
    messages.append({
        "role": "user",
        "content": [{"text": user_text}],
    })

    console.print(f"[dim]Thinking...[/dim]")

    response = client.converse(
        modelId=settings.bedrock_model_id,
        messages=messages,
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={
            "maxTokens": 200,  # Keep responses short for voice
            "temperature": 0.7,
        },
    )

    # Extract response text
    assistant_text = response["output"]["message"]["content"][0]["text"]

    # Append assistant message to history
    messages.append({
        "role": "assistant",
        "content": [{"text": assistant_text}],
    })

    console.print(f"[magenta]🤖 Response:[/magenta] {assistant_text}")
    return assistant_text


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
