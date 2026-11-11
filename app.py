"""Interface inicial para escrever e revisar um roteiro."""

import streamlit as st


st.set_page_config(page_title="NineCast", page_icon="🎙️")

st.title("🎙️ NineCast")
st.write("Transforme seus conteúdos de estudo em podcasts personalizados.")

st.subheader("Seu roteiro")
st.caption("Escreva ou cole o texto que deseja narrar e revise-o antes da geração.")
script = st.text_area(
    "Texto do episódio",
    height=300,
    placeholder="Comece escrevendo seu roteiro aqui...",
)

if script.strip():
    st.caption(f"{len(script.split())} palavras · {len(script)} caracteres")

st.info(
    "A seleção da voz e a geração de áudio serão adicionadas na próxima etapa. "
    "Por enquanto, copie seu roteiro antes de fechar ou atualizar a página."
)
