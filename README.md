# NineCast

Transforme seus conteúdos de estudo em episódios de podcast narrados localmente com clonagem de voz e preparação de roteiros com IA local.

## 1. Visão geral e proposta do NineCast

O **NineCast** é uma aplicação pessoal, gratuita e open-source que converte anotações, resumos e apostilas em PDF em episódios de podcast em formato MP3, utilizando clonagem de voz baseada em **Coqui XTTS-v2** e preparação opcional de roteiros via **Ollama** diretamente na **CPU**.

Todo o processamento — extração e limpeza de PDF, preparação de roteiro com IA local, normalização pré-TTS em português, condicionamento da voz, síntese neural por blocos, pós-processamento acústico e biblioteca de episódios — acontece **100% na sua máquina**, sem envio de textos ou gravações para serviços externos e com servidor local restrito a `127.0.0.1`.

### Destaques da versão `0.4.0`
- **Preparação de roteiros com IA local (`services/llm.py` + Ollama):** novo painel **✨ Preparar roteiro com IA local** com os modos **Roteiro didático** e **Resumo de revisão**, remoção automática de citações acadêmicas, divisão segura em seções, proibição estrita de cálculos/invenções, sinalização de cortes (`truncated`) e validação obrigatória de modelo local via loopback (`127.0.0.1` / `localhost` / `[::1]`).
- **Orquestração sequencial de memória RAM:** `voice.unload_model()` (`gc.collect()` + `malloc_trim(0)`) e `reserve_memory_for_llm()` liberam o XTTS-v2 (~3,4 GiB) antes de carregar o LLM no Ollama e forçam `keep_alive: 0` ao concluir, evitando que os dois modelos disputem os 16 GB de RAM ao mesmo tempo.
- **Maior fidelidade à voz de referência (preparo v3 + masterização transparente):** filtro passa-altas de `75 Hz` nas referências, ponderação do `speaker_embedding` pela duração útil de cada amostra, renormalização L2, poda de sobras curtas (`< 2 s`) no condicionamento de estilo e cadeia FFmpeg transparente (`highpass=f=80`, `acompressor` e `loudnorm` `LRA=9`, sem equalizadores fixos) em **192 kbps mono a 48 kHz**.
- **Base sólida das versões anteriores:** extração e limpeza de ruído em PDFs (PyMuPDF), normalização pré-TTS em PT-BR (`R$`, `%`, datas, ordinais, números, símbolos), 1 a 5 amostras `.wav`, títulos personalizados com tags ID3, duração `mm:ss` na biblioteca e lançador portátil Linux (`ninecast.sh`).

---

## 2. Pré-requisitos

- **Sistema operacional:** Linux x86_64 (validado em CPU Ryzen 5 PRO 5650U — 6 núcleos / 12 threads com 16 GB de RAM).
- **Python:** `3.12` (definido em `.python-version`).
- **Gerenciador de pacotes:** [uv](https://docs.astral.sh/uv/).
- **FFmpeg:** instalado no sistema (incluindo `ffprobe`) e disponível no `PATH`:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **Ollama *(opcional — necessário apenas para o recurso de preparação de roteiro com IA local)*:** instalado localmente e com um modelo leve em português baixado (testado com `qwen2.5:3b`):
  ```bash
  ollama pull qwen2.5:3b
  ```
- **Espaço livre em disco:** cerca de **2,5 GB** para `.venv` + XTTS-v2 (em `data/models/`) e mais **~2 GB** caso utilize o modelo `qwen2.5:3b` no Ollama.

---

## 3. Instalação e configuração do ambiente

1. Clone o repositório e acesse a pasta do projeto:
   ```bash
   git clone git@github.com:Empty7k/NineCast.git
   cd NineCast
   ```

2. Crie o ambiente virtual Python `3.12` e sincronize as dependências travadas no `uv.lock`:
   ```bash
   uv venv --python 3.12 .venv
   uv sync --frozen
   ```

3. *(Opcional)* Configure variáveis locais a partir do arquivo de exemplo:
   ```bash
   cp .env.example .env
   ```
   O arquivo `.env` é lido automaticamente por `config.py` e nunca é versionado. Principais variáveis:
   - `COQUI_TOS_AGREED=0` (mantenha `0` para exigir confirmação na interface ou `1` caso já tenha aceitado a licença não comercial CPML).
   - `DEFAULT_LANGUAGE=pt` (idioma padrão da síntese).
   - `MAX_TEXT_CHARS=50000` (limite máximo de caracteres por geração).
   - `XTTS_MAX_CHARS=220` (tamanho máximo de cada bloco enviado ao XTTS-v2).
   - `CPU_THREADS=6` (quantidade de threads físicas usadas pelo PyTorch e pelo Ollama na CPU).
   - `OLLAMA_BASE_URL=http://127.0.0.1:11434` (URL HTTP local em loopback do Ollama).
   - `OLLAMA_MODEL=qwen2.5:3b` (nome do modelo local no Ollama).
   - `LLM_NUM_CTX=4096` (janela em tokens. O perfil avançado usa até 2048).
   - `LLM_TIMEOUT_SECONDS=600` (tempo de espera por seção).
   - `LLM_MAX_SECTION_CHARS=3200` (limite de caracteres por seção enviada ao LLM).

---

## 4. Como executar o projeto

Para preparar roteiros, instale o Ollama e baixe o modelo desejado. O NineCast reutiliza um servidor ativo. Ao transformar um roteiro ou clicar em **Atualizar modelos locais**, ele inicia o Ollama em loopback sob demanda e encerra o processo que abriu ao terminar.

### Opção A — Lançador portátil (recomendado)
Na raiz do repositório, execute:
```bash
./ninecast.sh
```
O script valida a presença do `.venv` e inicia o Streamlit localmente em `http://127.0.0.1:8501`. Para encerrar o servidor, pressione `Ctrl+C` no terminal.

### Opção B — Atalho no menu de aplicativos do Linux
Para criar um atalho gráfico local em `~/.local/share/applications/NineCast.desktop` apontando automaticamente para o caminho atual do seu clone:
```bash
./ninecast.sh --install-desktop
```
- O instalador preserva um atalho existente para evitar sobrescritas acidentais.
- Se você mover a pasta do projeto no futuro, remova `~/.local/share/applications/NineCast.desktop` e execute `./ninecast.sh --install-desktop` novamente.

### Opção C — Execução manual via CLI
```bash
.venv/bin/streamlit run app.py
```

---

## 5. Fluxo básico de uso

1. **Selecionar ou cadastrar uma voz (Barra lateral):**
   - Utilize apenas gravações de voz que você tenha autorização explícita para clonar.
   - Marque a caixa **Li e aceito a licença não comercial do XTTS-v2**.
   - Se ainda não houver voz cadastrada, abra **➕ Adicionar nova voz autorizada**, informe o nome do perfil, selecione de **1 a 5 arquivos `.wav`** limpos (preferencialmente de 10 a 30 segundos no total, sem música de fundo ou ruído) e clique em **Cadastrar voz**.
   - Os arquivos enviados são higienizados em um diretório temporário e convertidos em latentes (`.latents.pt`) dentro de `data/voices/`.

2. **Preparar o roteiro (Aba `🎙️ Novo Episódio`):**
   - *(Opcional)* Informe o **Título do episódio**, usado para compor o nome do arquivo MP3 e gravar os metadados ID3.
   - Digite/cole o texto no campo **Texto do episódio**, **ou** importe um arquivo com **📄 Importar texto de PDF**.
   - *(Opcional)* Abra **✨ Preparar roteiro com IA local** e clique em **Atualizar modelos locais** para selecionar outro modelo instalado. Escolha entre **Roteiro didático (explicação para aula)** ou **Resumo de revisão** e clique em **Transformar roteiro com IA**. Revise o texto gerado no editor antes de gerar o áudio.
   - Acompanhe em tempo real os cartões de métricas com **Palavras**, **Blocos XTTS** (já considerando a expansão da normalização pré-TTS) e **Duração estimada**.

3. **Gerar e ouvir o episódio:**
   - Clique em **🎙️ Gerar Podcast**.
   - O painel de status exibirá o progresso bloco a bloco e a etapa final de codificação MP3.
   - Ao concluir, ouça o resultado no player integrado, confira as métricas (**Duração**, **Geração** e **RTF de síntese**) e clique em **⬇️ Baixar Podcast (.mp3)** se desejar salvar uma cópia fora de `data/podcasts/`.

4. **Gerenciar episódios gravados (Aba `📚 Minha Biblioteca`):**
   - Escolha qualquer episódio salvo em `data/podcasts/` no seletor para visualizar data, tamanho em MB e duração (`mm:ss`) e carregar seu player e botão de download sob demanda.
   - Para remover um episódio permanentemente, marque **Confirmar exclusão permanente** e clique em **Excluir episódio**.

---

## 6. Explicação resumida da lógica e dos principais componentes

- **[config.py](config.py):** Carrega variáveis do `.env` via `python-dotenv`, centraliza caminhos (`data/voices`, `data/podcasts`, `data/models`) e define parâmetros operacionais de áudio/TTS e LLM (`TTS_MODEL`, `DEFAULT_LANGUAGE`, `MAX_TEXT_CHARS`, `XTTS_MAX_CHARS`, `CPU_THREADS`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLM_NUM_CTX`, `LLM_MAX_SECTION_CHARS`).
- **[services/llm.py](services/llm.py):** Integra com o Ollama exclusivamente em HTTP loopback (`LocalOllama`, sem redirecionamentos), valida pesos locais via `/api/show`, remove citações acadêmicas (`remove_citations`), divide textos longos em seções respeitando parágrafos/frases (`_sections`), reserva RAM via `voice.reserve_memory_for_llm()`, sinaliza cortes por limite de tokens (`done_reason == "length"`) e garante o descarregamento do modelo (`keep_alive: 0`) em bloco `finally`.
- **[services/voice.py](services/voice.py):** Gerencia o carregamento sob demanda (*lazy loading*) e o descarregamento explícito (`unload_model()` com `gc.collect()` e `malloc_trim(0)`) do modelo XTTS-v2 em CPU com `threading.RLock`. No preparo das referências (`_prepare_reference` + `_extract_latents` + `_pool_reference_latents`), remove *DC offset*, aplica passa-altas de `75 Hz`, pondera o `speaker_embedding` pela duração útil de cada amostra com renormalização L2, poda apenas sobras finais curtas (`< 2 s`) no condicionamento de estilo e salva os latentes de forma atômica em `<voz>.latents.pt` (`0600`).
- **[services/tts_text.py](services/tts_text.py):** Normaliza expressões numéricas e símbolos antes do fatiamento para o TTS (`R$`, `%`, datas válidas com ano bissexto, ordinais `º`/`ª`, inteiros, decimais e símbolos `§`, `§§`, `&`), preservando códigos, versões e identificadores com zeros à esquerda.
- **[services/audio.py](services/audio.py):** Valida `MAX_TEXT_CHARS`, normaliza o texto para fala e divide-o por parágrafos e sentenças até `XTTS_MAX_CHARS` (`220` caracteres), protegendo abreviações em português e redistribuindo finais muito curtos (`< 20` caracteres). Na montagem (`_concatenate_wavs`), apara silêncios excedentes nas pontas preservando margem de `40 ms`, nivela o RMS ativo entre blocos (limitando ganho a `+6 dB` e pico a `-1 dBFS`), aplica fades de `12 ms` e insere pausas de `0,3 s` entre frases e `0,6 s` entre parágrafos. Exporta MP3 (`192 kbps`, mono, `48 kHz`) via FFmpeg com tags ID3 (`title` e `artist`) e cadeia transparente (`highpass=f=80`, `acompressor`, `loudnorm=I=-16:TP=-1.5:LRA=9`).
- **[services/content.py](services/content.py):** Extrai texto página a página com PyMuPDF (`pymupdf`), bloqueia PDFs vazios, corrompidos, protegidos por senha ou digitalizados sem OCR, remove cabeçalhos/rodapés repetidos, números de página isolados e URLs brutas, e une palavras hifenizadas entre linhas sem colar parágrafos distintos.
- **[services/library.py](services/library.py):** Lista arquivos `.mp3` regulares em `data/podcasts/` ordenados do mais recente ao mais antigo, consulta duração (`mm:ss`) via `ffprobe` com fallback seguro (`--:--`) e realiza exclusão validando nomes contra *path traversal* e links simbólicos.
- **[app.py](app.py) e [.streamlit/config.toml](.streamlit/config.toml):** Interface em Streamlit estruturada em cartões com tema escuro marítimo (*Atlântico Noturno*) e tratamento amigável de erros de validação e falta de memória (`MemoryError`).

---

## 7. Estrutura relevante do projeto

```text
NineCast/
├── .streamlit/
│   └── config.toml            # Tema escuro Atlântico Noturno e bind em 127.0.0.1
├── services/
│   ├── __init__.py
│   ├── audio.py               # Chunking, nivelamento RMS, fades WAV e masterização MP3 (FFmpeg)
│   ├── content.py             # Extração e limpeza de ruído de arquivos PDF (PyMuPDF)
│   ├── library.py             # Listagem com duração (ffprobe) e exclusão segura de episódios
│   ├── llm.py                 # Preparação local de roteiros com Ollama e gestão de RAM
│   ├── tts_text.py            # Normalização pré-TTS em português (R$, %, datas, números, símbolos)
│   └── voice.py               # Condicionamento v3, cache de latentes (.pt), unload e síntese XTTS-v2
├── tests/
│   ├── benchmark_llm.py       # Utilitário CLI para medir RAM, tokens/s e ciclo XTTS/Ollama
│   ├── benchmark_xtts.py      # Utilitário CLI para medir RAM, tempo de carga e RTF em CPU
│   ├── compare_xtts.py        # Utilitário CLI para comparar similaridade e timbre entre referências
│   ├── test_app.py            # Testes automatizados da interface via Streamlit AppTest
│   ├── test_audio_service.py  # Testes de normalização, chunking, nivelamento, fades e FFmpeg
│   ├── test_benchmark_llm.py  # Testes unitários da sequência e descarga do benchmark LLM
│   ├── test_content.py        # Testes de extração de PDF, limpeza de ruído, hífens e erros
│   ├── test_launcher.py       # Testes do lançador ninecast.sh e geração do atalho .desktop
│   ├── test_library.py        # Testes de ordenação, duração e exclusão segura da biblioteca
│   ├── test_llm_service.py    # Testes de loopback, validação local, seções e descarga Ollama
│   └── test_voice_service.py  # Testes de preparo v3, ponderação/L2, unload e síntese
├── data/                      # Diretório local ignorado pelo Git (criado em tempo de execução)
│   ├── models/                # Cache local dos pesos do XTTS-v2 e Hugging Face
│   ├── podcasts/              # Episódios MP3 gerados
│   └── voices/                # Perfis de voz (.latents.pt)
├── .env.example               # Exemplo de variáveis de configuração local
├── .gitignore                 # Regras de exclusão de dados locais, caches e ambientes
├── .python-version            # Versão do Python do projeto (3.12)
├── app.py                     # Aplicação principal Streamlit
├── config.py                  # Configurações e caminhos compartilhados
├── ninecast.sh                # Lançador portátil e instalador de atalho .desktop
├── pyproject.toml             # Metadados do projeto e dependências (v0.4.0)
└── uv.lock                    # Lockfile determinístico de pacotes
```

---

## 8. Testes e validações

Para executar toda a suíte de testes unitários e de interface:
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

Para executar módulos específicos:
```bash
.venv/bin/python tests/test_voice_service.py
.venv/bin/python tests/test_audio_service.py
.venv/bin/python tests/test_content.py
.venv/bin/python tests/test_library.py
.venv/bin/python tests/test_llm_service.py
.venv/bin/python tests/test_benchmark_llm.py
.venv/bin/python tests/test_launcher.py
.venv/bin/python tests/test_app.py
```

### Testes de integração real com XTTS-v2 (opcionais)
Se você possuir arquivos `.wav` de referência em `data/voices/` e os pesos do modelo em `data/models/`, pode rodar os testes reais fim a fim em CPU:
```bash
COQUI_TOS_AGREED=1 .venv/bin/python tests/test_voice_service.py --integration
COQUI_TOS_AGREED=1 .venv/bin/python tests/test_audio_service.py --integration
```

### Scripts de diagnóstico e benchmark
```bash
.venv/bin/python tests/benchmark_xtts.py --speaker-wav /caminho/para/referencia.wav --agree-license
.venv/bin/python tests/benchmark_llm.py --with-xttsCycle
.venv/bin/python tests/benchmark_llm.py --model qwen3:8b --thinking off
.venv/bin/python tests/benchmark_llm.py --model qwen3:14b --num-ctx 2048 --thinking off
```

O benchmark do Ollama amostra a memória disponível e o swap do sistema a cada 250 ms durante cada seção, além de registrar tempo, velocidade e motivo de término.

No Ryzen 5 PRO 5650U, o `qwen2.5:3b` foi a opção mais rápida para os dois textos de estudo (12,7–14,0 tokens/s). O `qwen3:8b` híbrido gerou 5,3–6,3 tokens/s e reescreveu a apostila em frases mais diretas. Ambos omitiram o termo *atomicidade*. A variante `qwen3:4b` testada exigia raciocínio, e o `qwen3:14b` experimental pressionou o swap mesmo com contexto de 2048 tokens. A interface verifica a capacidade de desativar o raciocínio e a RAM disponível para o 14B antes de gerar.

---

## 9. Observações, limitações e requisitos atuais

- **Desempenho em CPU:** Em CPU de 6 núcleos físicos (Ryzen 5 PRO 5650U com `CPU_THREADS=6`):
  - **XTTS-v2:** carregamento inicial em ~12–15 s, pico de ~3,4 GB de RAM e RTF médio de síntese ~2,5×.
  - **Ollama (`qwen2.5:3b`, `num_ctx=4096`):** ~14–17 tokens/s, ~2,05 GiB líquidos durante a geração e liberação imediata após o fim da chamada.
- **Limitações de modelos pequenos (3B) na preparação de roteiro:** Modelos compactos podem ocasionalmente parafrasear com perda de algum termo secundário ou simplificar demais um parágrafo. O texto gerado é sempre exibido para edição antes da síntese e **deve ser revisado** antes de gerar o episódio.
- **Limite de roteiro:** Roteiros são limitados por padrão a `50.000` caracteres (`MAX_TEXT_CHARS`) para preservar estabilidade de memória em CPU.
- **PDFs digitalizados (imagens):** A extração de PDF suporta apenas documentos com camada de texto selecionável. PDFs escaneados sem OCR retornam aviso orientando o usuário.
- **Heurísticas de limpeza e hifenização:** A união de palavras hifenizadas e a remoção de cabeçalhos/rodapés repetidos operam por heurística textual; recomenda-se revisar o texto extraído na caixa de roteiro antes de iniciar a síntese.
- **Player nativo do Streamlit:** O player HTML5 padrão carrega um episódio selecionado por vez na biblioteca para poupar RAM, mas ainda não oferece controles dedicados de velocidade de reprodução (`1.25x`/`1.5x`/`2x`) ou salto rápido de 15 segundos.
- **Ética e Licença:** Clone apenas a sua própria voz ou vozes com permissão expressa do titular. Os pesos do modelo **XTTS-v2** são distribuídos sob a [Coqui Public Model License (CPML)](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt), restrita a uso **não comercial**. Modelos baixados no Ollama seguem suas respectivas licenças (ex.: Apache 2.0 para `qwen2.5:3b`).

---

## 10. Futuras implementações previstas

- **Evolução da interface e do player de estudo (`v0.5.0+`):** migração da interface para camada dedicada com player completo (controle de velocidade `1.25×`/`1.5×`/`2×`, salto rápido de ±15 s), streaming de áudio direto do disco e fila em segundo plano com cancelamento real, mantendo `services/` desacoplado.
- **Suporte opcional a OCR local:** extração de texto para apostilas e livros digitalizados sem camada de texto selecionável.
- **Perfis e estilos avançados de roteiro:** suporte a múltiplos formatos de estudo (ex.: perguntas e respostas de revisão, glossário narrado).
