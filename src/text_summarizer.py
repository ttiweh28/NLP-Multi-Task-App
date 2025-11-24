from transformers import pipeline

text_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(text: str, max_length: int = 120, min_length: int = 40) -> str:
    """
    Summarize a long piece of text using a pre-trained BART model.
    """
    if not text.strip():
        return ""

    result = text_summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return result[0]["summary_text"]