# -*- coding: utf-8 -*-
import streamlit as st
from etf_metrics.core.chatbot import generate_advisory_response

FREE_MODELS = {
    "Qwen 3 (235B)": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen 3-Next (80B)": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Meta Llama 3.3 (70B)": "meta-llama/Llama-3.3-70B-Instruct",
    "DeepSeek-V3.2-Exp": "deepseek-ai/DeepSeek-V3.2-Exp"
}

DEFAULT_HF_TOKEN = "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"

def render_chatbot_ui():
    st.title("🤖 Financial Advisor AI")
    st.caption("Il tuo analista personale. Chiedi un'analisi su un titolo o ISIN.")

    with st.sidebar.expander("🧠 Impostazioni IA", expanded=True):
        hf_token = st.text_input("Hugging Face Token", value=DEFAULT_HF_TOKEN, type="password")

        selected_model_name = st.selectbox(
            "Scegli il Modello AI",
            options=list(FREE_MODELS.keys()),
            index=0
        )
        model_id = FREE_MODELS[selected_model_name]

        if hf_token:
            st.success(f"Modello attivo: {selected_model_name}")
        else:
            st.info("Inserisci il token per attivare l'IA.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ciao! Scrivimi un Ticker (es. NVDA) o un ISIN."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Analizza IE00B4L5Y983..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"L'IA ({selected_model_name}) sta analizzando i dati..."):
                response_text = generate_advisory_response(prompt, hf_token, model_id)
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})