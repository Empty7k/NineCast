"""Interface local do NineCast para criar e ouvir podcasts."""

import os
import tempfile
from pathlib import Path

import streamlit as st

import config
from services import audio, content, library, llm, voice


def prepare_script_with_ai() -> None:
    modes = {"Roteiro didático (explicação para aula)": "didatico", "Resumo de revisão": "resumo"}
    choice = st.radio("Modo de preparação", modes.keys(), key="script_mode")
    if st.button("Atualizar modelos locais", key="refresh_models"):
        try:
            with st.spinner("Consultando modelos do Ollama..."):
                st.session_state["local_models"] = llm.list_local_models(start_server=True)
            if not st.session_state["local_models"]:
                st.info("Nenhum modelo local encontrado. Baixe um modelo com ollama pull.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    models = {entry["name"]: entry["size_bytes"] for entry in st.session_state.get("local_models", [])}
    models.setdefault(config.OLLAMA_MODEL, None)
    if st.session_state.get("script_model") not in models:
        st.session_state["script_model"] = config.OLLAMA_MODEL
    selected_model = st.selectbox("Modelo local", models.keys(), key="script_model")
    description = llm.describe_model(selected_model, models[selected_model])
    st.caption(f"{description['tier']} · RAM {description['ram_estimada']} · CPU {description['velocidade_cpu']}")
    st.write(f"**Vantagens** · {description['vantagens']}")
    st.write(f"**Desvantagens** · {description['desvantagens']}")
    st.caption(description["recomendado_para"])
    if description["alerta_memoria"]:
        st.warning(description["alerta_memoria"])
    st.caption(
        "O Ollama roda neste computador. O roteiro preparado "
        "substitui o texto do editor. Confira os dados e revise a explicação antes de narrar."
    )
    if not st.button("Transformar roteiro com IA", key="prepare_script"):
        return
    source = st.session_state.get("script_text", "")
    if not source.strip():
        st.error("Escreva um texto ou importe um PDF antes de preparar o roteiro.")
        return
    if len(source) > config.MAX_TEXT_CHARS:
        st.error(f"O roteiro excede o limite de {config.MAX_TEXT_CHARS} caracteres. Reduza o texto e tente novamente.")
        return

    with st.status("Preparando roteiro com IA local...", expanded=True) as status:
        progress = st.progress(0)

        def section_processed(current: int, total: int) -> None:
            progress.progress(current / total, text=f"Seção {current} de {total} processada")

        try:
            result = llm.generate_study_script(
                source, mode=modes[choice], model=selected_model, progress_callback=section_processed,
            )
        except (ValueError, RuntimeError) as exc:
            status.update(label="Falha ao preparar o roteiro", state="error", expanded=True)
            st.error(str(exc))
            return
        status.update(label="Preparação concluída", state="complete", expanded=False)
    st.session_state["script_text"] = result["script"]
    if result["truncated"]:
        st.warning("Parte do roteiro atingiu o limite de saída e pode estar incompleta. Revise os trechos antes de narrar.")
    st.success("Roteiro preparado. Revise o texto no editor antes de gerar o áudio.")


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
        uploaded = st.file_uploader(
            "Arquivos de referência (.wav)",
            type=["wav"],
            accept_multiple_files=True,
            help="Selecione entre 1 e 5 amostras da mesma voz.",
        )
        register_voice = st.button("Cadastrar voz")

    license_accepted = st.checkbox(
        "Li e aceito a licença não comercial do XTTS-v2",
        help="No primeiro uso, os pesos do modelo podem ser baixados. As referências permanecem locais.",
    )
    st.markdown("[Consultar licença do modelo](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)")

    if register_voice:
        if not new_voice_name.strip() or not uploaded:
            st.error("Informe o nome da voz e selecione de 1 a 5 arquivos WAV.")
        elif len(uploaded) > 5:
            st.error("Selecione no máximo 5 arquivos WAV para uma voz.")
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
                        references = []
                        for index, sample in enumerate(uploaded, start=1):
                            reference = Path(directory) / f"referencia_{index:02d}.wav"
                            reference.write_bytes(sample.getvalue())
                            references.append(reference)
                        voice.get_or_create_voice_latents(new_voice_name, references)
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
        episode_title = st.text_input(
            "Título do episódio (opcional)",
            placeholder="Ex.: Introdução a bancos de dados",
        )
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
        with st.expander("✨ Preparar roteiro com IA local"):
            prepare_script_with_ai()
        script = st.text_area("Texto do episódio", height=300, placeholder="Comece escrevendo seu roteiro aqui...", key="script_text")
        if script.strip():
            normalized_script = audio.normalize_text_for_tts(script)
            words = len(script.split())
            chunks = audio.split_text_into_chunks(normalized_script) if normalized_script.strip() else []
            estimated_minutes = words / 150
            columns = st.columns(3)
            columns[0].metric("Palavras", words)
            columns[1].metric("Blocos XTTS", len(chunks))
            columns[2].metric("Duração estimada", f"~{estimated_minutes:.1f} min")
            if len(script) > config.MAX_TEXT_CHARS:
                st.warning(
                    f"O roteiro tem {len(script)} caracteres e excede o limite de "
                    f"{config.MAX_TEXT_CHARS}. Reduza o texto antes de gerar."
                )

    if st.button("🎙️ Gerar Podcast", type="primary"):
        if not script.strip():
            st.error("Escreva um texto para gerar o podcast.")
        elif len(script) > config.MAX_TEXT_CHARS:
            st.error(
                f"O roteiro excede o limite de {config.MAX_TEXT_CHARS} caracteres. "
                "Reduza o texto antes de iniciar a síntese."
            )
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

                  result = audio.generate_podcast(
                      script,
                      selected_voice,
                      progress_callback=update_progress,
                      episode_title=episode_title,
                  )
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
                                       file_name=path.name, mime="audio/mp3")
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
            st.caption(
                f"Modificado em {episode['created_at']} · {episode['size_mb']:.2f} MB · "
                f"{episode['duration']}"
            )
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
