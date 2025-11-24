from transformers import pipeline

text_generator = pipeline("text-generation", model="gpt2")


def generate_text(prompt: str, max_new_tokens: int = 120) -> str:
    """
    Generate text given a creative prompt using GPT-2.
    """
    if not prompt.strip():
        return ""

    result = text_generator(
        prompt,
        max_new_tokens=max_new_tokens,
    )
    return result[0]["generated_text"]