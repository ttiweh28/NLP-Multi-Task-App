import streamlit as st
from src.text_summarizer import summarize_text, get_summarization_model_names
from src.text_generator import generate_text, get_generation_model_names
from src.translator import translate_text, get_language_options


st.set_page_config(
    page_title="NLP Mini Multi-Task App",
    page_icon="🧠",
    layout="centered",
)


with st.sidebar:
    st.markdown("## NLP Mini Multi-Task App")
    st.markdown("This is a mini demo app using Hugging Face transformers.")

    # st.image("assets/logo.png", use_container_width=True)

    st.markdown("---")
    task = st.selectbox(
        "Choose a Task",
        [
            "📝 Summarization",
            "✍️ Text Generation",
            "🌐 Translation",
        ],
    )
    st.markdown("---")
    st.caption("Built with Streamlit + Hugging Face")



st.title("NLP Mini Multi-Task App")
st.markdown(
    "Using pre-trained transformer models for **summarization**, "
    "**creative text generation**, and **machine translation**."
)
st.markdown("---")


if task == "📝 Summarization":
    st.header("📝 Text Summarization")

    st.write(
        "Paste a long paragraph below and generate a concise summary "
        "using a pre-trained BART model."
    )

    text = st.text_area(
        "Input Text",
        height=200,
        placeholder="Enter a news article, blog paragraph, or any long text here...",
    )

    model_choice = st.selectbox(
        "Select summarization model",
        get_summarization_model_names()
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        max_len = st.slider("Max summary length", 50, 200, 120, step=10)
    with col2:
        min_len = st.slider("Min summary length", 20, 150, 40, step=10)

    st.markdown("")

    if st.button("Summarize"):
        if not text.strip():
            st.warning("Please enter text to summarize.")
        else:
            with st.spinner("Generating summary..."):
                summary = summarize_text(text,model_choice, max_length=max_len, min_length=min_len)
            st.subheader("Summary")
            st.write(summary)

    st.markdown("---")
    st.caption(
        "Powered by BART, Pegasus, and T5 models"
    )


elif task == "✍️ Text Generation":
    st.header("✍️ Creative Text Generation")

    st.write(
        "Provide a prompt and let GPT-2 continue the story or idea. "
        "You can control how long the continuation is."
    )

    prompt = st.text_area(
        "Enter a prompt",
        height=150,
        placeholder="For example: 'In the year 2085, humans and robots lived together in harmony, but one robot had a secret:'",
    )

    gen_model_choice = st.selectbox(
        "Select generation model",
        get_generation_model_names()
    )


    max_new_tokens = st.slider("Number of tokens", 30, 200, 120, step=10)

    st.markdown("")

    if st.button("Generate Text"):
        if not prompt.strip():
            st.warning("Please enter a prompt")
        else:
            with st.spinner("Generating text..."):
                output = generate_text(prompt,gen_model_choice, max_new_tokens=max_new_tokens)
            st.subheader("Generated Text")
            st.write(output)

    st.markdown("---")
    st.caption("Please be aware of load times. Also always review outputs.")


elif task == "🌐 Translation":
    st.header("🌐 Machine Translation")

    st.write(
        "Translate text between multiple language pairs using Helsinki-NLP "
        "translation models."
    )

    lang_options = get_language_options()
    lang_display = {label: code for code, label in lang_options}

    selected_label = st.selectbox(
        "Select language direction",
        [label for _, label in lang_options],
    )
    selected_code = lang_display[selected_label]

    text_to_translate = st.text_area(
        "Enter text to translate",
        height=150,
        placeholder="Enter text to translate...",
    )

    if st.button("Translate"):
        if not text_to_translate.strip():
            st.warning("Please enter text to translate.")
        else:
            with st.spinner(f"Translating ({selected_label})..."):
                translated = translate_text(text_to_translate, lang_pair=selected_code)
            st.subheader("Translated Output")
            st.write(translated)

    st.markdown("---")
    st.caption(
        "Backed by Helsinki-NLP models."
    )
