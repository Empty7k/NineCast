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
    if message := st.session_state.pop("voice_success", None):
        st.success(message)
    voices = voice.list_available_voices()
    pending_voice = st.session_state.pop("pending_voice", None)
    if pending_voice in voices:
        st.session_state["voice_selection"] = pending_voice
    if voices:
        if st.session_state.get("voice_selection") not in voices:
            st.session_state["voice_selection"] = voices[0]
        selected_voice = st.selectbox("Voz cadastrada", voices, key="voice_selection")
    else:
        selected_voice = None
        st.info("Nenhuma voz cadastrada. Adicione sua primeira voz autorizada acima.")

    with st.expander("➕ Adicionar nova voz autorizada"):
        new_voice_name = st.text_input("Nome da voz")
        uploaded = st.file_uploader("Arquivo de referência (.wav)", type=["wav"])
        register_voice = st.button("Cadastrar voz")

    license_accepted = st.checkbox(
        "Li e aceito a licença não comercial do XTTS-v2",
        help="No primeiro uso, os pesos do modelo podem ser baixados. As referências permanecem locais.",
    )
    st.markdown("[Consultar licença do modelo](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)")

    if register_voice:
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
                st.session_state["pending_voice"] = new_voice_name
                st.session_state["voice_success"] = f"Voz '{new_voice_name}' cadastrada com sucesso."
                st.rerun()
            except MemoryError:
                st.error("Memória insuficiente. Feche programas pesados e tente novamente.")
            except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                st.error(f"Não foi possível cadastrar a voz: {exc}")

tab_novo, tab_biblioteca = st.tabs(["🎙️ Novo Episódio", "📚 Minha Biblioteca"])

with tab_novo:
    with st.container(border=True):
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
            columns = st.columns(3)
            columns[0].metric("Palavras", words)
            columns[1].metric("Blocos XTTS", len(chunks))
            columns[2].metric("Duração estimada", f"~{estimated_minutes:.1f} min")

    if st.button("🎙️ Gerar Podcast", type="primary"):
        if not script.strip():
            st.error("Escreva um texto para gerar o podcast.")
        elif not selected_voice:
            st.error("Cadastre e selecione uma voz antes de gerar o podcast.")
        elif not license_accepted:
            st.error("Leia e aceite a licença do XTTS-v2 antes de gerar o podcast.")
        else:
            with st.status("Preparando a narração...", expanded=True) as status:
                progress = st.progress(0, text="Preparando a geração...")
                try:
                  os.environ["COQUI_TOS_AGREED"] = "1"

                  def update_progress(current: int, total: int) -> None:
                      progress.progress(current / total, text=f"Trecho {current} de {total} sintetizado.")
                      if current == total:
                          status.update(label="Finalizando e exportando o MP3...")

                  result = audio.generate_podcast(script, selected_voice, progress_callback=update_progress)
                  st.session_state["last_episode"] = result
                  status.update(label="Podcast gerado com sucesso!", state="complete", expanded=False)
                except MemoryError:
                  status.update(label="Falha na geração", state="error")
                  st.error("Memória insuficiente. Feche programas pesados e tente novamente.")
                except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                  status.update(label="Falha na geração", state="error")
                  st.error(f"Não foi possível gerar o podcast: {exc}")

    if result := st.session_state.get("last_episode"):
        path = Path(result["output_path"])
        if path.is_file():
            try:
                with path.open("rb") as episode:
                    episode_bytes = episode.read()
                with st.container(border=True):
                    st.subheader("Seu podcast")
                    st.audio(episode_bytes, format="audio/mp3")
                    st.download_button("⬇️ Baixar Podcast (.mp3)", data=episode_bytes,
                                       file_name="meu_podcast.mp3", mime="audio/mp3")
                    columns = st.columns(3)
                    columns[0].metric("Duração", f"{result['audio_duration']:.1f} s")
                    columns[1].metric("Geração", f"{result['total_seconds']:.1f} s")
                    columns[2].metric("RTF de síntese", f"{result['rtf']:.2f}×")
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
        st.caption(f"Total de episódios: {len(episodes)}")
        selected_episode = st.selectbox("Escolha um episódio", [e["filename"] for e in episodes])
        episode = next(e for e in episodes if e["filename"] == selected_episode)
        name = episode["filename"]
        with st.container(border=True):
            st.subheader(name)
            st.caption(f"Modificado em {episode['created_at']} · {episode['size_mb']:.2f} MB")
            try:
                st.audio(str(episode["path"]), format="audio/mp3")
                with episode["path"].open("rb") as audio_file:
                    st.download_button("⬇️ Baixar MP3", data=audio_file, file_name=name,
                                       mime="audio/mp3", key=f"download_{name}")
            except OSError as exc:
                st.error(f"Não foi possível abrir o episódio: {exc}")
            st.divider()
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
