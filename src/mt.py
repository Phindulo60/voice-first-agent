"""
Machine Translation (NLLB-200)
isiZulu ↔ English translation using Meta's No Language Left Behind model.

Runs locally — no API dependency.
Uses the distilled 600M model (fits in ~2GB RAM, good quality for Zulu).
"""

from rich.console import Console

from src.config import settings

console = Console()

# NLLB model + tokenizer (lazy singletons)
_model = None
_tokenizer = None

# NLLB language codes
ZULU = "zul_Latn"
ENGLISH = "eng_Latn"


def _load_model():
    """Load NLLB model and tokenizer."""
    global _model, _tokenizer

    if _model is None:
        console.print(f"[dim]Loading NLLB translation model: {settings.nllb_model}...[/dim]")

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        _tokenizer = AutoTokenizer.from_pretrained(settings.nllb_model)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.nllb_model,
            torch_dtype=torch.float32,
        )

        # Move to device
        device = settings.mt_device
        if device != "cpu":
            import torch
            if device == "mps" and torch.backends.mps.is_available():
                _model = _model.to("mps")
            elif device == "cuda" and torch.cuda.is_available():
                _model = _model.to("cuda")

        console.print("[green]✓ NLLB translation model loaded[/green]")

    return _model, _tokenizer


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text between any NLLB-supported language pair.

    Args:
        text: Input text to translate
        src_lang: Source language code (e.g., 'zul_Latn', 'eng_Latn')
        tgt_lang: Target language code

    Returns:
        Translated text
    """
    model, tokenizer = _load_model()

    # Set source language
    tokenizer.src_lang = src_lang

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get target language token ID
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    # Generate translation
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=256,
    )

    # Decode
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result


def zulu_to_english(text: str) -> str:
    """Translate isiZulu text to English."""
    console.print(f"[dim]Translating zu→en...[/dim]")
    result = translate(text, src_lang=ZULU, tgt_lang=ENGLISH)
    console.print(f"[cyan]🔄 zu→en:[/cyan] {result}")
    return result


def english_to_zulu(text: str) -> str:
    """Translate English text to isiZulu."""
    console.print(f"[dim]Translating en→zu...[/dim]")
    result = translate(text, src_lang=ENGLISH, tgt_lang=ZULU)
    console.print(f"[cyan]🔄 en→zu:[/cyan] {result}")
    return result


# Allow running standalone for testing
if __name__ == "__main__":
    console.print("[bold]Translation Test (NLLB-200) — isiZulu ↔ English[/bold]\n")
    console.print("[dim]Commands: 'zu' for Zulu→English, 'en' for English→Zulu, 'q' to quit[/dim]\n")

    direction = "zu"
    while True:
        try:
            line = input(f"\n[{direction}→{'en' if direction == 'zu' else 'zu'}] Text: ").strip()
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break
            if line.lower() in ("zu", "en"):
                direction = line.lower()
                console.print(f"[dim]Switched to {'Zulu→English' if direction == 'zu' else 'English→Zulu'}[/dim]")
                continue

            if direction == "zu":
                zulu_to_english(line)
            else:
                english_to_zulu(line)
        except KeyboardInterrupt:
            break

    console.print("\n[dim]Done.[/dim]")
