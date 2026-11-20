# NineCast

Transforme seus conteúdos de estudo, anotações e PDFs em episódios de podcast narrados com clonagem de voz — **100% local, gratuito e open-source**.

## 1. Visão geral e proposta

O **NineCast** converte textos e apostilas em PDF em episódios de podcast em formato MP3 diretamente no seu computador (em **CPU**, sem exigir placa de vídeo dedicada).

Todo o fluxo — extração e limpeza de PDF, preparação opcional de roteiro com IA local, normalização de texto em português, clonagem de voz, masterização acústica e biblioteca de episódios — acontece **localmente**, sem envio de dados ou gravações para a nuvem e com servidor restrito a `127.0.0.1`.

### O que o NineCast oferece na versão `0.5.0`
- **Dois motores locais de clonagem de voz à escolha:**
  - **XTTS-v2 *(padrão — mais rápido)*:** gera episódios mais rápido em CPU e aceita de 1 a 5 amostras `.wav` sem precisar de transcrição.
  - **Qwen3-TTS 0.6B *(opcional — voz mais natural e fiel)*:** licença Apache-2.0, roda em ambiente isolado (`.venv-qwen-tts/`), aceita de 1 a 3 amostras `.wav` e faz **transcrição automática local** da referência via `faster-whisper`.
- **Preparação de roteiros com IA local (Ollama sob demanda):** transforma textos densos ou PDFs em **Roteiro didático** ou **Resumo de revisão**. O NineCast inicia e encerra `ollama serve` automaticamente quando necessário, lista os modelos locais instalados e exibe perfil, uso estimado de RAM, prós e contras de cada opção.
- **Gestão inteligente de memória:** libera automaticamente o motor de voz da RAM antes de rodar a IA local ou o processo isolado do Qwen3-TTS, evitando disputa de memória.
- **Limpeza de PDFs e normalização pré-TTS (PT-BR):** extrai texto com PyMuPDF removendo cabeçalhos/rodapés repetidos, números de página soltos e URLs, e converte valores (`R$`), porcentagens (`%`), datas, ordinais, números e símbolos (`§`, `&`) para fala natural antes da síntese.
- **Áudio tratado e biblioteca local:** nivelamento entre blocos, micro-fades, masterização transparente via FFmpeg (`192 kbps`, mono, `48 kHz`), tags ID3 (`title`/`artist`) e biblioteca com duração (`mm:ss`), player sob demanda e exclusão com confirmação.

---

## 2. Requisitos de sistema

O NineCast foi projetado para rodar em **CPU** (sem necessidade de GPU dedicada):

| Recurso | Mínimo recomendado *(XTTS-v2)* | Recomendado *(Qwen3-TTS + IA local)* |
|---|---|---|
| **Sistema operacional** | Linux x86_64 | Linux x86_64 |
| **Processador (CPU)** | 4 núcleos físicos ou mais | 6 núcleos físicos ou mais |
| **Memória RAM** | 8 GB | 16 GB *(modelos 8B+ exigem RAM livre extra)* |
| **Espaço livre em disco** | ~3 GB *(ambiente + XTTS-v2)* | ~8–12 GB *(+ Qwen3-TTS, Whisper e modelos Ollama)* |

### Ferramentas necessárias
- **Python `3.12`** (gerenciado automaticamente pelo `uv`).
- **[uv](https://docs.astral.sh/uv/)** (gerenciador de ambientes e pacotes Python).
- **FFmpeg** (incluindo `ffprobe`) instalado no sistema.
- **[Ollama](https://ollama.com/)** *(opcional — apenas se quiser usar a preparação de roteiros com IA local)*.

---

## 3. Tutorial de instalação passo a passo

Siga os passos abaixo para deixar tudo pronto do zero:

### Passo 1 — Instalar dependências do sistema (`ffmpeg`, `uv` e opcionalmente `ollama`)

- **FFmpeg:**
  ```bash
  # Ubuntu / Debian / Linux Mint
  sudo apt update && sudo apt install -y ffmpeg

  # Arch Linux / CachyOS / Manjaro
  sudo pacman -S ffmpeg

  # Fedora
  sudo dnf install ffmpeg
  ```
- **uv:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  *(Reinicie o terminal ou adicione `~/.local/bin` ao `PATH` se o comando `uv` não for reconhecido imediatamente.)*

- **Ollama *(opcional — para preparar roteiros com IA local)*:**
  Instale pelo gerenciador da sua distro (ex.: `sudo pacman -S ollama`) ou pelo instalador oficial em [ollama.com](https://ollama.com/download/linux).

### Passo 2 — Clonar o repositório e instalar o ambiente principal

```bash
git clone https://github.com/Empty7k/NineCast.git
cd NineCast
uv venv --python 3.12 .venv
uv sync --frozen
```

Com isso, o NineCast já está pronto para funcionar com o motor **XTTS-v2**.

### Passo 3 — *(Opcional)* Instalar o motor Qwen3-TTS 0.6B + Transcrição Automática

Se quiser usar o motor de voz mais natural (**Qwen3-TTS 0.6B**, Apache-2.0) com transcrição automática via `faster-whisper`, execute:

```bash
./ninecast.sh --install-qwen-tts
```

Esse comando cria um ambiente virtual isolado em `.venv-qwen-tts/` sem alterar nem conflitar com o ambiente principal `.venv/`. Na primeira utilização, os pesos do `Qwen3-TTS-12Hz-0.6B-Base` (~2,3 GB) e do `Whisper small` INT8 (~240 MB) são baixados localmente.

### Passo 4 — *(Opcional)* Baixar um modelo local no Ollama

Se instalou o Ollama e deseja usar o painel **✨ Preparar roteiro com IA local**, baixe ao menos um modelo local:

```bash
# Opção leve e rápida (recomendada como padrão para a maioria das máquinas)
ollama pull qwen2.5:3b

# Opção intermediária (mais lenta e exige mais RAM, útil para textos técnicos)
ollama pull qwen3:8b
```

> Não é necessário deixar o `ollama serve` aberto manualmente: o NineCast reutiliza um servidor ativo ou inicia um processo temporário em loopback (`OLLAMA_NO_CLOUD=1`) apenas durante a geração e o encerra ao terminar.

### Passo 5 — *(Opcional)* Ajustar configurações locais (`.env`)

Caso queira personalizar o número de threads da sua CPU, o motor padrão ou o modelo do Ollama:

```bash
cp .env.example .env
```

Principais opções disponíveis no `.env` (carregado automaticamente por `config.py` e ignorado pelo Git):
- `CPU_THREADS=6` — ajuste para a quantidade de **núcleos físicos** da sua CPU (ex.: `4`, `6`, `8`).
- `TTS_ENGINE=xtts` — motor inicial selecionado na interface (`xtts` ou `qwen3_06b`).
- `DEFAULT_LANGUAGE=pt` — idioma padrão da síntese e transcrição.
- `MAX_TEXT_CHARS=50000` — limite máximo de caracteres por episódio.
- `XTTS_MAX_CHARS=220` / `QWEN_TTS_MAX_CHARS=220` — tamanho máximo por bloco de síntese.
- `QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base` — modelo do Qwen3-TTS.
- `WHISPER_MODEL=small` — modelo `faster-whisper` usado na auto-transcrição.
- `OLLAMA_BASE_URL=http://127.0.0.1:11434` — endereço local (apenas loopback) do Ollama.
- `OLLAMA_MODEL=qwen2.5:3b` — modelo padrão exibido na IA local.
- `LLM_NUM_CTX=4096`, `LLM_MAX_SECTION_CHARS=3200`, `LLM_TIMEOUT_SECONDS=600` — contexto, tamanho de seção e tempo máximo por seção do LLM.
- `COQUI_TOS_AGREED=0` — mantenha `0` para confirmar na interface ou `1` se já aceitou a licença CPML do XTTS-v2.

---

## 4. Como executar o projeto

### Opção A — Lançador portátil (recomendado)
Na raiz do repositório:
```bash
./ninecast.sh
```
O NineCast abrirá localmente em `http://127.0.0.1:8501`. Pressione `Ctrl+C` no terminal para encerrar.

### Opção B — Criar atalho no menu de aplicativos do Linux
```bash
./ninecast.sh --install-desktop
```
Gera `~/.local/share/applications/NineCast.desktop` apontando para o caminho atual do seu clone (preservando um atalho já existente; se mover a pasta do projeto, remova o `.desktop` antigo e execute o comando novamente).

### Opção C — Execução manual via CLI
```bash
.venv/bin/streamlit run app.py
```

---

## 5. Fluxo básico de uso

1. **Escolher motor e cadastrar voz (Barra lateral):**
   - Utilize apenas gravações de voz que você tenha autorização explícita para clonar.
   - Escolha entre **XTTS-v2** (mais rápido) e **Qwen3-TTS 0.6B** (voz mais natural e fiel; clique em **ℹ️ Sobre este motor** para ver prós, contras e licença).
   - Abra **➕ Adicionar nova voz autorizada**:
     - No **XTTS-v2:** informe o nome, envie de **1 a 5 arquivos `.wav`** limpos e marque a caixa da licença não comercial CPML.
     - No **Qwen3-TTS 0.6B:** informe o nome e envie de **1 a 3 arquivos `.wav`** (o 1º define o ritmo e a transcrição; mais arquivos nem sempre melhoram a clonagem — um único clipe limpo pode soar melhor). Clique em **✨ Transcrever automaticamente** para revisar o texto ou deixe o campo em branco para transcrever automaticamente ao cadastrar.

2. **Preparar o roteiro (Aba `🎙️ Novo Episódio`):**
   - *(Opcional)* Defina um **Título do episódio** (usado no nome do arquivo MP3 e nas tags ID3).
   - Escreva/cole o texto diretamente ou abra **📄 Importar texto de PDF** para extrair o conteúdo limpo de uma apostila.
   - *(Opcional)* Abra **✨ Preparar roteiro com IA local**, clique em **Atualizar modelos locais** se quiser trocar o modelo instalado no Ollama, escolha entre **Roteiro didático** ou **Resumo de revisão** e clique em **Transformar roteiro com IA**. Revise sempre o texto retornado no editor antes de narrar.

3. **Gerar e ouvir o podcast:**
   - Confira as métricas (**Palavras**, **Blocos XTTS/Qwen** e **Duração estimada**) e clique em **🎙️ Gerar Podcast**.
   - Acompanhe o progresso por bloco; ao finalizar, ouça no player integrado ou clique em **⬇️ Baixar Podcast (.mp3)**.

4. **Gerenciar episódios (Aba `📚 Minha Biblioteca`):**
   - Selecione qualquer episódio salvo em `data/podcasts/` para ver data, tamanho em MB e duração (`mm:ss`), ouvir sob demanda, baixar ou excluir com confirmação.

---

## 6. Explicação resumida da lógica e dos principais componentes

- **[config.py](config.py):** Carrega `.env` via `python-dotenv`, centraliza caminhos (`data/voices`, `data/podcasts`, `data/models`) e concentra todos os parâmetros de áudio, voz e IA.
- **[services/voice.py](services/voice.py):** Gerencia o **XTTS-v2** com carregamento sob demanda e descarregamento explícito (`unload_model()` com `gc.collect()` + `malloc_trim(0)` e `reserve_memory_for_llm()`). Prepara referências com remoção de DC, corte de pausas longas, filtro passa-altas de `75 Hz`, ponderação do `speaker_embedding` pela duração com renormalização L2 e poda de sobras curtas (`< 2 s`) no condicionamento de estilo, salvando latentes em `<voz>.latents.pt` (`0600`).
- **[services/qwen_tts.py](services/qwen_tts.py) e [services/qwen_tts_worker.py](services/qwen_tts_worker.py):** Gerenciam o **Qwen3-TTS 0.6B** e a transcrição local com **faster-whisper** dentro do ambiente isolado `.venv-qwen-tts/`, usando um único subprocesso por episódio com protocolo JSON isolado (`stdout` redirecionado para `stderr`) e encerramento garantido ao final. Suportam perfis de 1 a 3 WAVs (ICL no 1º clipe + combinação ponderada de embeddings preservando a norma L2 média quando há extras).
- **[services/llm.py](services/llm.py):** Integra com o **Ollama** exclusivamente em HTTP loopback (`LocalOllama`, sem redirecionamentos), inicia e encerra `ollama serve` sob demanda (`OLLAMA_NO_CLOUD=1`), lista modelos locais (`/api/tags`), valida pesos locais (`/api/show`), bloqueia variantes de raciocínio obrigatório, envia `think: false` quando suportado, remove citações acadêmicas, divide textos longos em seções e sinaliza cortes (`done_reason == "length"`).
- **[services/tts_text.py](services/tts_text.py):** Normaliza moedas (`R$`), porcentagens (`%`), datas válidas (incluindo ano bissexto), ordinais (`º`/`ª`), números e símbolos (`§`, `§§`, `&`) para português falado preservando códigos, versões e IPs.
- **[services/audio.py](services/audio.py):** Expõe o catálogo de motores (`TTS_ENGINES` / `describe_tts_engine`) e orquestra `generate_podcast()` para ambos os motores (`xtts` e `qwen3_06b`), aplicando normalização pré-TTS, fatiamento, trim de pontas (`40 ms`), nivelamento RMS sem *clipping*, micro-fades (`12 ms`), pausas entre frases (`0,3 s`) e parágrafos (`0,6 s`) e exportação MP3 (`192 kbps`, mono, `48 kHz`, tags ID3) via FFmpeg (`highpass=f=80`, `acompressor`, `loudnorm=I=-16:TP=-1.5:LRA=9`).
- **[services/content.py](services/content.py) e [services/library.py](services/library.py):** Extração/limpeza de PDFs (PyMuPDF) e gerenciamento seguro da biblioteca local com duração via `ffprobe`.
- **[app.py](app.py) e [.streamlit/config.toml](.streamlit/config.toml):** Interface em Streamlit com tema escuro *Atlântico Noturno*, resumos curtos e detalhes recolhidos por motor/modelo.

---

## 7. Estrutura relevante do projeto

```text
NineCast/
├── .streamlit/
│   └── config.toml              # Tema escuro Atlântico Noturno e bind em 127.0.0.1
├── services/
│   ├── __init__.py
│   ├── audio.py                 # Catálogo TTS, fatiamento, montagem WAV e masterização MP3
│   ├── content.py               # Extração e limpeza de ruído de arquivos PDF (PyMuPDF)
│   ├── library.py               # Listagem com duração (ffprobe) e exclusão segura de MP3s
│   ├── llm.py                   # Ollama sob demanda, catálogo multi-modelo e gestão de RAM
│   ├── qwen_tts.py              # Perfis 1–3 WAVs, auto-transcrição e cliente do worker Qwen
│   ├── qwen_tts_worker.py       # Subprocesso isolado do Qwen3-TTS 0.6B e faster-whisper
│   ├── tts_text.py              # Normalização pré-TTS em PT-BR (R$, %, datas, números)
│   └── voice.py                 # XTTS-v2 (preparo v3), cache de latentes (.pt) e unload
├── tests/
│   ├── benchmark_llm.py         # Benchmark CLI do Ollama (RAM, swap, tokens/s)
│   ├── benchmark_xtts.py        # Benchmark CLI do XTTS-v2 em CPU
│   ├── compare_xtts.py          # Comparativo de similaridade entre referências XTTS
│   ├── test_app.py              # Testes de interface via Streamlit AppTest
│   ├── test_audio_service.py    # Testes de catálogo TTS, normalização, montagem e FFmpeg
│   ├── test_benchmark_llm.py    # Testes unitários do fluxo do benchmark LLM
│   ├── test_content.py          # Testes de extração e limpeza de PDFs
│   ├── test_launcher.py         # Testes do ninecast.sh (--install-desktop / --install-qwen-tts)
│   ├── test_library.py          # Testes de biblioteca e duração
│   ├── test_llm_service.py      # Testes de loopback, auto-start Ollama, thinking e RAM
│   ├── test_qwen_tts.py         # Testes de perfis 1–3 WAVs, worker Qwen e Whisper
│   └── test_voice_service.py    # Testes de condicionamento v3, unload e síntese XTTS
├── data/                        # Dados locais ignorados pelo Git (criado em execução)
│   ├── models/                  # Cache local de modelos (XTTS, Whisper, Hugging Face)
│   ├── podcasts/                # Episódios MP3 gerados
│   └── voices/                  # Perfis XTTS (.latents.pt) e Qwen (qwen/<nome>/)
├── .env.example                 # Modelo de variáveis de configuração local
├── .gitignore                   # Exclusão de dados privados, caches e ambientes virtuais
├── .python-version              # Versão do Python do projeto (3.12)
├── app.py                       # Interface principal Streamlit
├── config.py                    # Configurações compartilhadas
├── ninecast.sh                  # Lançador portátil e instaladores (--install-desktop / --install-qwen-tts)
├── pyproject.toml               # Metadados e dependências do ambiente principal (v0.5.0)
└── uv.lock                      # Lockfile determinístico de dependências
```

---

## 8. Testes e validações

Para executar toda a suíte automatizada (**108 testes**, rápida e sem exigir XTTS, Qwen ou Ollama em execução):
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### Testes de integração real com XTTS-v2 (opcionais)
Se houver amostras `.wav` em `data/voices/` e os pesos do XTTS-v2 em `data/models/`:
```bash
COQUI_TOS_AGREED=1 .venv/bin/python tests/test_voice_service.py --integration
COQUI_TOS_AGREED=1 .venv/bin/python tests/test_audio_service.py --integration
```

### Scripts de diagnóstico e benchmark em CPU
```bash
.venv/bin/python tests/benchmark_xtts.py --speaker-wav /caminho/para/referencia.wav --agree-license
.venv/bin/python tests/benchmark_llm.py --with-xttsCycle
```

---

## 9. Observações, limitações e licenças

- **Desempenho em CPU:** A síntese neural local em CPU é mais lenta que o tempo real — o **XTTS-v2** costuma ser ~2× mais rápido, enquanto o **Qwen3-TTS 0.6B** prioriza naturalidade e fidelidade do timbre.
- **Modelos de IA local:** Modelos compactos (`3B`) são rápidos para uso diário, mas podem simplificar termos técnicos; modelos maiores (`8B`–`14B`) exigem significativamente mais RAM livre (modelos `14B` são experimentais em máquinas de 16 GB e só iniciam com margem de memória disponível). Revise sempre o roteiro antes de narrar.
- **PDFs digitalizados:** Apenas PDFs com camada de texto selecionável são suportados atualmente (sem OCR integrado).
- **Player atual:** O player nativo do Streamlit ainda não possui controle dedicado de velocidade (`1.25×`/`1.5×`/`2×`) nem botão de cancelar geração em andamento.
- **Ética e Licenciamento:**
  - Clone apenas a sua própria voz ou vozes com autorização explícita do titular.
  - **XTTS-v2:** sujeito à [Coqui Public Model License (CPML)](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) (uso **não comercial**).
  - **Qwen3-TTS 0.6B:** licença **Apache-2.0**.
  - **faster-whisper / modelos Whisper:** licença **MIT**.
  - **Modelos Ollama:** seguem a licença de cada modelo baixado (ex.: Apache-2.0 para `qwen2.5:3b` e `qwen3:8b`).

---

## 10. Próximas implementações previstas

- **Nova interface visual e player de estudo (`v0.6.0`):** migração da camada visual (saindo do Streamlit para uma interface web dedicada mantendo `services/` intacto) com player completo (controle de velocidade `1.25×`/`1.5×`/`2×`, salto rápido de ±15 s), streaming de áudio direto do disco e fila em segundo plano com botão real de cancelar.
- **Suporte opcional a OCR local:** leitura de PDFs e apostilas digitalizadas sem camada de texto.
- **Conversão automática de amostras de voz:** aceitar formatos como `.mp3`, `.m4a` e `.ogg` direto no cadastro de voz.
