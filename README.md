# NineCast

Transforme seus conteúdos de estudo em episódios de podcast narrados localmente com clonagem de voz.

## 1. Visão geral e proposta do NineCast

O **NineCast** é uma aplicação pessoal, gratuita e open-source que converte anotações, resumos e apostilas em PDF em episódios de podcast em formato MP3, utilizando clonagem de voz baseada em **Coqui XTTS-v2** diretamente na **CPU**.

Todo o processamento — extração de PDF, condicionamento da voz, síntese neural por blocos, pós-processamento acústico e biblioteca de episódios — acontece **100% na sua máquina**, sem envio de textos ou gravações para serviços externos e com servidor local restrito a `127.0.0.1`.

### Destaques da versão `0.2.0`
- **Clonagem de voz local com cache inteligente:** extrai e armazena apenas os vetores latentes (`.latents.pt`) em `data/voices/`, validados por hash SHA-256 das referências, descartando o upload temporário após o cadastro.
- **Importação local de PDF:** extrai o texto das páginas via PyMuPDF, une palavras hifenizadas em quebras de linha e preserva a divisão de parágrafos para revisão antes da síntese.
- **Pipeline de áudio com pós-processamento FFmpeg:** divide o texto respeitando parágrafos, sentenças e abreviações em português, insere pausas naturais entre frases e parágrafos e aplica equalização e normalização de loudness (`loudnorm`) na exportação MP3.
- **Biblioteca local sob demanda:** lista os episódios gerados em `data/podcasts/`, carregando o player e o botão de download apenas para o episódio selecionado (economizando memória RAM) e exigindo confirmação antes de excluir arquivos.
- **Lançador portátil e integração com o desktop Linux:** script `ninecast.sh` com gerador opcional de atalho `.desktop` adaptável a qualquer diretório em que o projeto for clonado.

---

## 2. Pré-requisitos

- **Sistema operacional:** Linux x86_64 (validado em CPU de 6 núcleos / 12 threads com 16 GB de RAM).
- **Python:** `3.12` (definido em `.python-version`).
- **Gerenciador de pacotes:** [uv](https://docs.astral.sh/uv/).
- **FFmpeg:** instalado no sistema e disponível no `PATH`:
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
   - `MAX_TEXT_CHARS=50000` (limite de referência para textos longos).
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
   - Se ainda não houver voz cadastrada, abra **➕ Adicionar nova voz autorizada**, informe o nome do perfil, selecione um arquivo `.wav` limpo (preferencialmente de 10 a 30 segundos, sem música de fundo ou ruído) e clique em **Cadastrar voz**.
   - O arquivo enviado é processado em um diretório temporário e convertido em latentes (`.latents.pt`) dentro de `data/voices/`.

2. **Preparar o roteiro (Aba `🎙️ Novo Episódio`):**
   - Digite ou cole o texto diretamente no campo **Texto do episódio**, **ou**
   - Abra **📄 Importar texto de PDF**, selecione um PDF com camada de texto e clique em **Extrair texto do PDF**. O conteúdo extraído preenche o editor para que você possa revisar títulos, rodapés ou siglas antes de narrar.
   - Acompanhe em tempo real os cartões de métricas com **Palavras**, **Blocos XTTS** e **Duração estimada**.

3. **Gerar e ouvir o episódio:**
   - Clique em **🎙️ Gerar Podcast**.
   - O painel de status exibirá o progresso bloco a bloco e a etapa final de codificação MP3.
   - Ao concluir, ouça o resultado no player integrado, confira as métricas (**Duração**, **Geração** e **RTF de síntese**) e clique em **⬇️ Baixar Podcast (.mp3)** se desejar salvar uma cópia fora de `data/podcasts/`.

4. **Gerenciar episódios gravados (Aba `📚 Minha Biblioteca`):**
   - Escolha qualquer episódio salvo em `data/podcasts/` no seletor para carregar seu player e o botão de download sob demanda.
   - Para remover um episódio permanentemente, marque **Confirmar exclusão permanente** e clique em **Excluir episódio**.

---

## 6. Explicação resumida da lógica e dos principais componentes

- **[config.py](config.py):** Carrega variáveis do `.env` via `python-dotenv`, centraliza caminhos (`data/voices`, `data/podcasts`, `data/models`) e define parâmetros operacionais (`TTS_MODEL`, `DEFAULT_LANGUAGE`, `XTTS_MAX_CHARS`, `CPU_THREADS`).
- **[services/voice.py](services/voice.py):** Gerencia o carregamento sob demanda (*lazy loading*) do modelo XTTS-v2 em CPU com `threading.RLock`. No preparo da referência (`_prepare_reference`), converte o áudio para mono, remove silêncios longos preservando pausas naturais de dicção (`< 0,35 s`) e gera os tensores `gpt_cond_latent` e `speaker_embedding`, salvos de forma atômica em `<voz>.latents.pt` (`0600`). Na síntese (`synthesize_speech`), gera blocos WAV mono em `24 kHz` (`PCM_16`).
- **[services/audio.py](services/audio.py):** Divide o texto por parágrafos e sentenças até `XTTS_MAX_CHARS` (`220` caracteres), protegendo abreviações em português (`Sr.`, `Dra.`, `Prof.`, `etc.`, `pág.`, `fig.`) e redistribuindo finais muito curtos (`< 20` caracteres). Concatena os WAVs via streaming com pausas de `0,3 s` entre frases do mesmo parágrafo e `0,6 s` na troca de parágrafos, e codifica o MP3 final (`128 kbps`, mono, `24 kHz`) via FFmpeg aplicando filtros de equalização e normalização (`loudnorm=I=-16:TP=-1.5:LRA=11`).
- **[services/content.py](services/content.py):** Extrai texto página a página com PyMuPDF (`pymupdf`), bloqueia PDFs vazios, corrompidos, protegidos por senha ou digitalizados sem OCR e une palavras separadas por hífen no fim de linha sem colar parágrafos distintos.
- **[services/library.py](services/library.py):** Lista arquivos `.mp3` regulares em `data/podcasts/` ordenados do mais recente ao mais antigo e realiza exclusão segura validando nomes contra *path traversal* e links simbólicos.
- **[app.py](app.py) e [.streamlit/config.toml](.streamlit/config.toml):** Interface em Streamlit estruturada em cartões com tema escuro marítimo (*Atlântico Noturno*) e tratamento amigável de erros de validação e falta de memória (`MemoryError`).

---

## 7. Estrutura relevante do projeto

```text
NineCast/
├── .streamlit/
│   └── config.toml            # Tema escuro Atlântico Noturno e bind em 127.0.0.1
├── services/
│   ├── __init__.py
│   ├── audio.py               # Chunking de texto, concatenação WAV e exportação MP3 (FFmpeg)
│   ├── content.py             # Extração e limpeza de texto de arquivos PDF (PyMuPDF)
│   ├── library.py             # Listagem e exclusão segura de episódios em data/podcasts/
│   └── voice.py               # Condicionamento de voz, cache de latentes (.pt) e síntese XTTS-v2
├── tests/
│   ├── benchmark_xtts.py      # Utilitário CLI para medir RAM, tempo de carga e RTF em CPU
│   ├── compare_xtts.py        # Utilitário CLI para comparar conjuntos de referências WAV
│   ├── test_app.py            # Testes automatizados da interface via Streamlit AppTest
│   ├── test_audio_service.py  # Testes de divisão de texto, silêncios, FFmpeg e pipeline de áudio
│   ├── test_content.py        # Testes de extração de PDF, acentuação, hífens e erros
│   ├── test_launcher.py       # Testes do lançador ninecast.sh e geração do atalho .desktop
│   ├── test_library.py        # Testes de ordenação e exclusão segura da biblioteca
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
├── pyproject.toml             # Metadados do projeto e dependências (v0.2.0)
└── uv.lock                    # Lockfile determinístico de pacotes
```

---

## 8. Testes e validações

Para executar toda a suíte de testes unitários e de interface (rápida, sem exigir download do modelo XTTS-v2):
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
- **PDFs digitalizados (imagens):** A extração de PDF suporta apenas documentos com camada de texto selecionável. PDFs escaneados sem OCR retornam aviso orientando o usuário.
- **Heurística de hifenização:** A união automática de palavras hifenizadas entre linhas pode juntar palavras compostas que coincidam exatamente com uma quebra de linha do PDF; recomenda-se revisar o texto extraído na caixa de roteiro antes de iniciar a síntese.
- **Player nativo do Streamlit:** O player HTML5 padrão exibe um episódio selecionado por vez na biblioteca para poupar RAM, mas ainda não oferece controles dedicados de velocidade de reprodução (`1.25x`/`1.5x`/`2x`) ou salto rápido de 15 segundos.
- **Ética e Licença:** Clone apenas a sua própria voz ou vozes com permissão expressa do titular. Os pesos do modelo **XTTS-v2** são distribuídos sob a [Coqui Public Model License (CPML)](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt), restrita a uso **não comercial**.

---

## 10. Futuras implementações previstas

- **Refinamento avançado de áudio:** aprimoramento da cadeia de pós-processamento para reduzir artefatos de borda entre blocos, melhorar a estabilidade tímbrica em textos longos e permitir ajustes finos de cadência.
- **Integração com IA / LLM local:** reescrita assistida e sumarização local de textos brutos ou apostilas em PDF para transformá-los automaticamente em roteiros didáticos com linguagem natural de podcast antes da narração.
- **Evolução da interface e do player de estudo:** controles dedicados de velocidade de reprodução, avanço/retrocesso rápido, organização por temas/cadernos de estudo e migração futura da camada visual mantendo `services/` desacoplado.
