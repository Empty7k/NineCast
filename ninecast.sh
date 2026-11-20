#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--install-qwen-tts" ]]; then
    if [[ "$#" -ne 1 ]]; then
        printf 'Uso: %s --install-qwen-tts\n' "$0" >&2
        exit 2
    fi
    if ! command -v uv >/dev/null; then
        printf 'uv não encontrado no PATH. Instale o uv para preparar o Qwen3-TTS.\n' >&2
        exit 1
    fi
    QWEN_ENV="$PROJECT_DIR/.venv-qwen-tts"
    if [[ ! -x "$QWEN_ENV/bin/python" ]]; then
        uv venv --python 3.12 "$QWEN_ENV"
    fi
    uv pip install --python "$QWEN_ENV/bin/python" --index https://download.pytorch.org/whl/cpu \
        'torch==2.8.0+cpu' 'torchaudio==2.8.0+cpu'
    uv pip install --python "$QWEN_ENV/bin/python" 'qwen-tts==0.1.1' soundfile numpy 'faster-whisper==1.2.1'
    printf 'Ambiente do Qwen3-TTS instalado em %s\n' "$QWEN_ENV"
    exit 0
fi
if [[ "${1:-}" == "--install-desktop" ]]; then
    if [[ "$#" -ne 1 ]]; then
        printf 'Uso: %s --install-desktop\n' "$0" >&2
        exit 2
    fi
    if [[ "$PROJECT_DIR" == *$'\n'* || "$PROJECT_DIR" == *$'\r'* ]]; then
        printf 'O caminho do projeto não pode conter quebras de linha.\n' >&2
        exit 1
    fi
    APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p -- "$APP_DIR"
    DESKTOP_FILE="$APP_DIR/NineCast.desktop"
    # Não substitui um atalho existente sem a intervenção do usuário.
    if ! (set -o noclobber; : > "$DESKTOP_FILE") 2>/dev/null; then
        printf 'Não foi possível criar %s; verifique se já existe ou se há permissão.\n' "$DESKTOP_FILE" >&2
        exit 1
    fi
    EXEC_PATH=${PROJECT_DIR//\\/\\\\}
    EXEC_PATH=${EXEC_PATH//\"/\\\"}
    EXEC_PATH=${EXEC_PATH//\$/\\\$}
    EXEC_PATH=${EXEC_PATH//\`/\\\`}
    EXEC_PATH=${EXEC_PATH//%/%%}
    DESKTOP_PATH=${PROJECT_DIR//\\/\\\\}
    printf '%s\n' '[Desktop Entry]' 'Type=Application' 'Name=NineCast' \
        'Comment=Podcasts de estudo narrados por IA' \
        "Exec=\"$EXEC_PATH/ninecast.sh\"" "Path=$DESKTOP_PATH" \
        'Icon=audio-headphones' 'Terminal=true' 'Categories=Education;AudioVideo;Audio;' > "$DESKTOP_FILE"
    chmod 0755 "$DESKTOP_FILE"
    printf 'Atalho instalado em: %s\n' "$DESKTOP_FILE"
    exit 0
fi
if [[ ! -x "$PROJECT_DIR/.venv/bin/streamlit" ]]; then
    printf 'Streamlit não encontrado no ambiente virtual. Execute: uv sync --locked\n' >&2
    exit 1
fi
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/streamlit" run "$PROJECT_DIR/app.py" "$@"
