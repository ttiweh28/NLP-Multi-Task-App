from transformers import pipeline
LANGUAGE_MODELS = {
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-sw": "Helsinki-NLP/opus-mt-en-sw",
    "sw-en": "Helsinki-NLP/opus-mt-sw-en",
}

LANGUAGE_LABELS = {
    "en-es": "English → Spanish",
    "es-en": "Spanish → English",
    "en-fr": "English → French",
    "fr-en": "French → English",
    "en-sw": "English → Swahili",
    "sw-en": "Swahili → English",
}

# Cache of loaded pipelines 
translators = {}


def _get_translator(lang_pair: str):
    if lang_pair not in LANGUAGE_MODELS:
        raise ValueError(f"Unsupported language pair: {lang_pair}")

    if lang_pair not in translators:
        model_name = LANGUAGE_MODELS[lang_pair]
        translators[lang_pair] = pipeline("translation", model=model_name)

    return translators[lang_pair]


def translate_text(text: str, lang_pair: str) -> str:
    """
    Translate text according to the selected language pair.
    """
    if not text.strip():
        return ""

    translator = _get_translator(lang_pair)
    result = translator(text)
    return result[0]["translation_text"]


def get_language_options():
    """
    Returns a list of (code, label) tuples for the UI dropdown.
    """
    return [(code, LANGUAGE_LABELS[code]) for code in LANGUAGE_MODELS.keys()]
