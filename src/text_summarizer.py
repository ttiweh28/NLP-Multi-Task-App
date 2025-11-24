from transformers import pipeline

SUMMARIZATION_MODELS = {
    "BART (Large CNN)": "facebook/bart-large-cnn",
    "PEGASUS (XSum)": "google/pegasus-xsum",
    "T5 Small": "t5-small",
}

_loaded_summarizers = {}


def summarize_text(text: str, model_name: str,
                   max_length: int = 120, min_length: int = 40) -> str:

    if not text.strip():
        return ""

    if model_name not in _loaded_summarizers:
        _loaded_summarizers[model_name] = pipeline(
            "summarization",
            model=SUMMARIZATION_MODELS[model_name],
        )

    summarizer = _loaded_summarizers[model_name]
    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    return result[0]["summary_text"]


def get_summarization_model_names():
    return list(SUMMARIZATION_MODELS.keys())
