import streamlit as st

def download_button(label: str, text: str, filename: str):
    st.download_button(
        label=label,
        data=text,
        file_name=filename,
        mime="text/plain",
    )

def copy_to_clipboard_button(text: str, label="Copy to Clipboard"):
    st.code(text)
    st.button(label, on_click=st.session_state.update({"copy_output": text}))
