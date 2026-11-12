"""Interface local do NineCast para criar e ouvir podcasts."""

import os
import tempfile
from pathlib import Path

import streamlit as st

import config
from services import audio, content, library, voice


st.set_page_config(page_title="NineCast", page_icon="🎙️")
st.title("🎙️ NineCast")
st.write("Transforme seus conteúdos de estudo em podcasts personalizados.")

with st.sidebar:
    st.header("Voz do episódio")
    st.caption("Use apenas gravações de voz que você tenha autorização para utilizar.")
    license_accepted = st.checkbox(
        "Li e aceito a licença não comercial do XTTS-v2",
        help="No primeiro uso, os pesos do modelo podem ser baixados. As referências permanecem locais.",
    )
    st.markdown("[Consultar licença do modelo](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)")

    with st.expander("➕ Adicionar nova voz autorizada"):
        new_voice_name = st.text_input("Nome da voz")
        uploaded = st.file_uploader("Arquivo de referência (.wav)", type=["wav"])
        if st.button("Cadastrar voz"):
            if not new_voice_name.strip() or uploaded is None:
                st.error("Informe o nome da voz e selecione um arquivo WAV.")
            elif not license_accepted:
                st.error("Leia e aceite a licença do XTTS-v2 antes de cadastrar a voz.")
            elif new_voice_name in voice.list_available_voices():
                st.error("Já existe uma voz com esse nome. Escolha outro nome.")
            else:
                try:
                    # O upload é temporário: somente os latentes permanecem no perfil.
                    config.VOICES_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
                    os.environ["COQUI_TOS_AGREED"] = "1"
                    with st.spinner("Preparando a voz. O primeiro carregamento pode demorar..."):
                        with tempfile.TemporaryDirectory(prefix="ninecast_upload_", dir=config.VOICES_DIR) as directory:
                            reference = Path(directory) / "referencia.wav"
                            reference.write_bytes(uploaded.getvalue())
                            voice.get_or_create_voice_latents(new_voice_name, [reference])
                    st.session_state["voice_selection"] = new_voice_name
                    st.session_state["voice_success"] = f"Voz '{new_voice_name}' cadastrada com sucesso."
                    st.rerun()
                except MemoryError:
                    st.error("Memória insuficiente. Feche programas pesados e tente novamente.")
                except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                    st.error(f"Não foi possível cadastrar a voz: {exc}")

    if message := st.session_state.pop("voice_success", None):
        st.success(message)
    voices = voice.list_available_voices()
    if voices:
        if st.session_state.get("voice_selection") not in voices:
            st.session_state["voice_selection"] = voices[0]
        selected_voice = st.selectbox("Voz cadastrada", voices, key="voice_selection")
    else:
        selected_voice = None
        st.info("Nenhuma voz cadastrada. Adicione sua primeira voz autorizada acima.")

tab_novo, tab_biblioteca = st.tabs(["🎙️ Novo Episódio", "📚 Minha Biblioteca"])

with tab_novo:
    st.subheader("Seu roteiro")
    st.caption("Escreva ou cole o texto que deseja narrar e revise-o antes da geração.")
    with st.expander("📄 Importar texto de PDF"):
        pdf = st.file_uploader("Documento PDF", type=["pdf"])
        st.caption("A extração é local. O texto importado substituirá o roteiro atual; revise-o antes de gerar áudio.")
        if st.button("Extrair texto do PDF"):
            if pdf is None:
                st.error("Selecione um documento PDF.")
            else:
                try:
                    st.session_state["script_text"] = content.extract_pdf_text(pdf.getvalue())
                    st.success("Texto extraído. Revise o roteiro abaixo.")
                except (ValueError, RuntimeError) as exc:
                    st.error(f"Não foi possível extrair o PDF: {exc}")
    script = st.text_area("Texto do episódio", height=300, placeholder="Comece escrevendo seu roteiro aqui...", key="script_text")
    if script.strip():
        words = len(script.split())
        chunks = audio.split_text_into_chunks(script)
        estimated_minutes = words / 150
        st.caption(f"{words} palavras · {len(script)} caracteres · Estimativa de ~{len(chunks)} blocos XTTS "
                   f"(~{estimated_minutes:.1f} min de áudio, a 150 palavras/min)")

    if st.button("🎙️ Gerar Podcast", type="primary"):
        if not script.strip():
            st.error("Escreva um texto para gerar o podcast.")
        elif not selected_voice:
            st.error("Cadastre e selecione uma voz antes de gerar o podcast.")
        elif not license_accepted:
            st.error("Leia e aceite a licença do XTTS-v2 antes de gerar o podcast.")
        else:
            progress = st.progress(0, text="Preparando a geração...")
            status = st.empty()
            try:
                os.environ["COQUI_TOS_AGREED"] = "1"

                def update_progress(current: int, total: int) -> None:
                    progress.progress(current / total, text=f"Trecho {current} de {total} sintetizado.")
                    if current == total:
                        status.info("Finalizando e exportando o MP3...")

                status.info("Carregando o modelo e preparando a narração...")
                result = audio.generate_podcast(script, selected_voice, progress_callback=update_progress)
                st.session_state["last_episode"] = result
                status.success("Podcast gerado com sucesso!")
            except MemoryError:
                status.empty()
                st.error("Memória insuficiente. Feche programas pesados e tente novamente.")
            except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                status.empty()
                st.error(f"Não foi possível gerar o podcast: {exc}")

    if result := st.session_state.get("last_episode"):
        path = Path(result["output_path"])
        if path.is_file():
            try:
                with path.open("rb") as episode:
                    episode_bytes = episode.read()
                st.subheader("Seu podcast")
                st.audio(episode_bytes, format="audio/mp3")
                st.download_button("⬇️ Baixar Podcast (.mp3)", data=episode_bytes,
                                   file_name="meu_podcast.mp3", mime="audio/mp3")
                st.caption(f"Duração: {result['audio_duration']:.1f} s · "
                           f"Tempo de geração: {result['total_seconds']:.1f} s · "
                           f"RTF de síntese: {result['rtf']:.2f}×")
                st.caption(f"Salvo localmente em: {path}")
            except OSError as exc:
                st.error(f"Não foi possível abrir o podcast salvo: {exc}")
        else:
            st.warning("O arquivo do último podcast não está mais disponível no caminho salvo.")

with tab_biblioteca:
    st.subheader("Minha biblioteca")
    if message := st.session_state.pop("library_message", None):
        st.success(message)
    episodes = library.list_episodes()
    if not episodes:
        st.info("Nenhum episódio gravado ainda. Crie seu primeiro podcast na aba 'Novo Episódio'!")
    else:
        st.write(f"Total de episódios: {len(episodes)}")
        selected_episode = st.selectbox("Escolha um episódio para ouvir ou baixar", [e["filename"] for e in episodes])
        for episode in episodes:
            name = episode["filename"]
            with st.expander(name):
                st.caption(f"Modificado em {episode['created_at']} · {episode['size_mb']:.2f} MB")
                if name == selected_episode:
                    try:
                        st.audio(str(episode["path"]), format="audio/mp3")
                        with episode["path"].open("rb") as audio_file:
                            st.download_button("⬇️ Baixar MP3", data=audio_file, file_name=name,
                                               mime="audio/mp3", key=f"download_{name}")
                    except OSError as exc:
                        st.error(f"Não foi possível abrir o episódio: {exc}")
                confirm = st.checkbox("Confirmar exclusão permanente", key=f"confirm_{name}")
                if st.button("Excluir episódio", key=f"delete_{name}", disabled=not confirm):
                    try:
                        deleted = library.delete_episode(name)
                        if deleted:
                            last = st.session_state.get("last_episode")
                            if last and Path(last["output_path"]).absolute() == episode["path"]:
                                del st.session_state["last_episode"]
                            st.session_state["library_message"] = f"Episódio '{name}' excluído."
                        else:
                            st.session_state["library_message"] = "O episódio já não estava disponível."
                        st.rerun()
                    except (ValueError, OSError) as exc:
                        st.error(f"Não foi possível excluir o episódio: {exc}")
