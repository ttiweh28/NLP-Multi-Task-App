from transformers import pipeline

GENERATION_MODELS = {
    "GPT-2": "gpt2",
    "GPT-Neo 1.3B": "EleutherAI/gpt-neo-1.3B",
}

_loaded_generators = {}


def generate_text(prompt: str, model_name: str,
                  max_new_tokens: int = 120) -> str:

    if not prompt.strip():
        return ""

    if model_name not in _loaded_generators:
        _loaded_generators[model_name] = pipeline(
            "text-generation",
            model=GENERATION_MODELS[model_name],
        )

    generator = _loaded_generators[model_name]
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens
    )
    return result[0]["generated_text"]


def get_generation_model_names():
    return list(GENERATION_MODELS.keys())
