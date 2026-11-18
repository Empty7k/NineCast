# NineCast

Transforme seus conteúdos de estudo em episódios de podcast narrados localmente com clonagem de voz.

## 1. Visão geral e proposta do NineCast

O **NineCast** é uma aplicação pessoal, gratuita e open-source que converte anotações, resumos e apostilas em PDF em episódios de podcast em formato MP3, utilizando clonagem de voz baseada em **Coqui XTTS-v2** diretamente na **CPU**.

Todo o processamento — extração e limpeza de PDF, normalização pré-TTS em português, condicionamento da voz, síntese neural por blocos, pós-processamento acústico e biblioteca de episódios — acontece **100% na sua máquina**, sem envio de textos ou gravações para serviços externos e com servidor local restrito a `127.0.0.1`.

### Destaques da versão `0.3.0`
- **Masterização vocal e montagem sem artefatos:** remoção de *DC offset* e normalização de pico (`-1 dBFS`) nas referências de voz; aparo inteligente de silêncios nas pontas (margem de `40 ms`), nivelamento RMS suave entre blocos com proteção contra *clipping*, micro-fades de `12 ms` e cadeia FFmpeg com passa-altas (`80 Hz`), equalização suave, compressor dinâmico (`acompressor`) e `loudnorm` (`LRA=9`) exportando MP3 a **192 kbps mono em 48 kHz**.
- **Normalização pré-TTS em português brasileiro:** converte automaticamente valores em reais (`R$`), porcentagens (`%`), datas válidas (`DD/MM/AAAA` e `DD/MM`, com ano bissexto e `"primeiro"` no dia `1`), ordinais (`º`/`ª`), números inteiros/decimais e símbolos de estudo (`§`, `§§`, `&`), preservando versões, IPs e códigos alfanuméricos.
- **Limpeza de ruído na extração de PDF:** além de unir palavras hifenizadas entre linhas e preservar parágrafos via PyMuPDF, remove numeração solta de páginas, URLs brutas e cabeçalhos/rodapés repetidos em documentos multipágina.
- **Usabilidade e biblioteca aprimoradas:** cadastro de voz aceitando de **1 a 5 amostras `.wav`**, título opcional do episódio (com nome de arquivo sanitizado e tags ID3 `title`/`artist`), bloqueio preventivo para textos acima de `MAX_TEXT_CHARS` (`50.000` caracteres) e exibição da duração (`mm:ss` via `ffprobe`) na biblioteca local sob demanda.
- **Lançador portátil e integração com o desktop Linux:** script `ninecast.sh` com gerador opcional de atalho `.desktop` adaptável a qualquer diretório em que o projeto for clonado.

---

## 2. Pré-requisitos

- **Sistema operacional:** Linux x86_64 (validado em CPU de 6 núcleos / 12 threads com 16 GB de RAM).
- **Python:** `3.12` (definido em `.python-version`).
- **Gerenciador de pacotes:** [uv](https://docs.astral.sh/uv/).
- **FFmpeg:** instalado no sistema (incluindo `ffprobe`) e disponível no `PATH`:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **Espaço livre em disco:** cerca de **2,5 GB** para o ambiente virtual e os pesos do modelo XTTS-v2 (baixados automaticamente em `data/models/` na primeira execução).

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
   O arquivo `.env` é lido automaticamente por `config.py` e nunca é versionado. Nele você pode ajustar:
   - `COQUI_TOS_AGREED=0` (mantenha `0` para exigir confirmação na interface ou `1` caso já tenha aceitado a licença não comercial CPML).
   - `DEFAULT_LANGUAGE=pt` (idioma padrão da síntese).
   - `MAX_TEXT_CHARS=50000` (limite máximo de caracteres por geração).
   - `XTTS_MAX_CHARS=220` (tamanho máximo de cada bloco enviado ao XTTS-v2).
   - `CPU_THREADS=6` (quantidade de threads físicas usadas pelo PyTorch na CPU).

---

## 4. Como executar o projeto

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
   - Digite ou cole o texto diretamente no campo **Texto do episódio**, **ou**
   - Abra **📄 Importar texto de PDF**, selecione um PDF com camada de texto e clique em **Extrair texto do PDF**. O conteúdo extraído (já sem cabeçalhos/rodapés repetidos, números de página soltos ou URLs brutas) preenche o editor para revisão antes de narrar.
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

- **[config.py](config.py):** Carrega variáveis do `.env` via `python-dotenv`, centraliza caminhos (`data/voices`, `data/podcasts`, `data/models`) e define parâmetros operacionais (`TTS_MODEL`, `DEFAULT_LANGUAGE`, `MAX_TEXT_CHARS`, `XTTS_MAX_CHARS`, `CPU_THREADS`).
- **[services/voice.py](services/voice.py):** Gerencia o carregamento sob demanda (*lazy loading*) do modelo XTTS-v2 em CPU com `threading.RLock`. No preparo da referência (`_prepare_reference`), converte o áudio para mono, remove *DC offset*, encurta silêncios longos preservando pausas naturais de dicção (`< 0,35 s`), normaliza o pico para `-1 dBFS` e gera os tensores `gpt_cond_latent` e `speaker_embedding`, salvos de forma atômica em `<voz>.latents.pt` (`0600`). Na síntese (`synthesize_speech`), gera blocos WAV mono em `24 kHz` (`PCM_16`).
- **[services/tts_text.py](services/tts_text.py):** Normaliza expressões numéricas e símbolos antes do fatiamento para o TTS (`R$`, `%`, datas válidas com ano bissexto, ordinais `º`/`ª`, inteiros, decimais e símbolos `§`, `§§`, `&`), preservando códigos, versões e identificadores com zeros à esquerda.
- **[services/audio.py](services/audio.py):** Valida `MAX_TEXT_CHARS`, normaliza o texto para fala e divide-o por parágrafos e sentenças até `XTTS_MAX_CHARS` (`220` caracteres), protegendo abreviações em português e redistribuindo finais muito curtos (`< 20` caracteres). Na montagem (`_concatenate_wavs`), apara silêncios excedentes nas pontas preservando margem de `40 ms`, nivela o RMS ativo entre blocos (limitando o ganho a `+6 dB` e o pico a `-1 dBFS`), aplica fades de `12 ms` e insere pausas de `0,3 s` entre frases e `0,6 s` entre parágrafos. Exporta o MP3 final (`192 kbps`, mono, `48 kHz`) via FFmpeg com tags ID3 (`title` e `artist`) e filtros `highpass=f=80`, equalização, `acompressor` e `loudnorm=I=-16:TP=-1.5:LRA=9`.
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
│   ├── tts_text.py            # Normalização pré-TTS em português (R$, %, datas, números, símbolos)
│   └── voice.py               # Condicionamento de voz, cache de latentes (.pt) e síntese XTTS-v2
├── tests/
│   ├── benchmark_xtts.py      # Utilitário CLI para medir RAM, tempo de carga e RTF em CPU
│   ├── compare_xtts.py        # Utilitário CLI para comparar conjuntos de referências WAV
│   ├── test_app.py            # Testes automatizados da interface via Streamlit AppTest
│   ├── test_audio_service.py  # Testes de normalização, chunking, nivelamento, fades e FFmpeg
│   ├── test_content.py        # Testes de extração de PDF, limpeza de ruído, hífens e erros
│   ├── test_launcher.py       # Testes do lançador ninecast.sh e geração do atalho .desktop
│   ├── test_library.py        # Testes de ordenação, duração e exclusão segura da biblioteca
│   └── test_voice_service.py  # Testes de preparo de referência, cache de latentes e síntese
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
├── pyproject.toml             # Metadados do projeto e dependências (v0.3.0)
└── uv.lock                    # Lockfile determinístico de pacotes
```

---

## 8. Testes e validações

Para executar toda a suíte de testes unitários e de interface (**54 testes**, rápida e sem exigir download do modelo XTTS-v2):
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

Para executar módulos específicos:
```bash
.venv/bin/python tests/test_voice_service.py
.venv/bin/python tests/test_audio_service.py
.venv/bin/python tests/test_content.py
.venv/bin/python tests/test_library.py
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
```

---

## 9. Observações, limitações e requisitos atuais

- **Desempenho em CPU:** Em um processador de 6 núcleos físicos (ex.: Ryzen 5 PRO 5650U com `CPU_THREADS=6`), o carregamento inicial do modelo leva ~12–15 s, o pico de memória fica em torno de **3,4 GB de RAM** e o *Real-Time Factor* (RTF) médio de síntese varia entre **1,7× e 2,5×** (cerca de 2 segundos de processamento para cada 1 segundo de fala gerada).
- **Limite de roteiro:** Roteiros são limitados por padrão a `50.000` caracteres (`MAX_TEXT_CHARS`) para preservar estabilidade de memória em CPU.
- **PDFs digitalizados (imagens):** A extração de PDF suporta apenas documentos com camada de texto selecionável. PDFs escaneados sem OCR retornam aviso orientando o usuário.
- **Heurísticas de limpeza e hifenização:** A união de palavras hifenizadas e a remoção de cabeçalhos/rodapés repetidos operam por heurística textual; recomenda-se revisar o texto extraído na caixa de roteiro antes de iniciar a síntese.
- **Player nativo do Streamlit:** O player HTML5 padrão carrega um episódio selecionado por vez na biblioteca para poupar RAM, mas ainda não oferece controles dedicados de velocidade de reprodução (`1.25x`/`1.5x`/`2x`) ou salto rápido de 15 segundos.
- **Ética e Licença:** Clone apenas a sua própria voz ou vozes com permissão expressa do titular. Os pesos do modelo **XTTS-v2** são distribuídos sob a [Coqui Public Model License (CPML)](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt), restrita a uso **não comercial**.

---

## 10. Futuras implementações previstas

- **Integração com IA / LLM local (`v0.4.0`):** reescrita assistida e sumarização local de textos brutos ou apostilas em PDF para transformá-los automaticamente em roteiros didáticos com linguagem natural de podcast antes da narração.
- **Evolução da interface e do player de estudo (`v0.5.0+`):** controles dedicados de velocidade de reprodução, avanço/retrocesso rápido, fila com cancelamento em segundo plano e migração da camada visual mantendo `services/` desacoplado.
- **Suporte opcional a OCR local:** extração de texto para apostilas e livros digitalizados sem camada de texto selecionável.
