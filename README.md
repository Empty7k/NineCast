# NineCast

Aplicativo pessoal para transformar conteúdos de estudo em podcasts narrados com
voz autorizada. A interface permite cadastrar uma voz, escrever um roteiro e
gerar um podcast em MP3 localmente. O texto não é salvo automaticamente.

## Preparação

Requisitos: Linux x86-64, Python 3.12, [uv](https://docs.astral.sh/uv/) e FFmpeg
com suporte a MP3. O ambiente usa versões de PyTorch para CPU.

No diretório do projeto, instale as dependências fixadas no lock:

```bash
uv sync --locked
```

O `uv` pode obter uma versão gerenciada do Python 3.12 se ela ainda não estiver
disponível. O arquivo `.python-version` seleciona a versão para o projeto.

## Executar

```bash
uv run --locked streamlit run app.py
```

Abra o endereço local mostrado no terminal. A aplicação escuta somente em
`127.0.0.1`. Na barra lateral, aceite a licença do XTTS-v2 e cadastre uma voz
autorizada com um WAV; depois, escreva o roteiro e selecione **Gerar Podcast**.
O primeiro carregamento pode demorar e, se os pesos não estiverem no cache,
exigirá um download. O player e o download aparecem após a geração; o MP3 é
salvo em `data/podcasts/`. A duração exibida antes de gerar é uma estimativa
baseada em 150 palavras por minuto, não o tempo de processamento.

O texto e o último resultado exibido permanecem na sessão da página; copie o
roteiro antes de atualizá-la ou fechá-la. O MP3 salvo continua no disco após
encerrar a sessão. Para encerrar o servidor, pressione `Ctrl+C` no terminal.

Na aba **Minha Biblioteca**, consulte os MP3s em `data/podcasts/`, reproduza-os
ou faça download. A data exibida é a última modificação do arquivo. A exclusão
exige confirmação e remove o arquivo permanentemente do disco.

Em **Importar texto de PDF**, selecione um documento e clique em **Extrair texto
do PDF**. O conteúdo substitui o roteiro atual e pode ser revisado antes da
geração. A extração ocorre localmente, sem salvar o PDF. Documentos digitalizados
sem camada de texto exigem OCR, que ainda não está disponível; a ordem de leitura
de documentos com várias colunas pode exigir revisão manual.

## Privacidade e dados

Não inclua gravações, documentos, roteiros pessoais, podcasts ou credenciais no
Git. O `.gitignore` exclui as pastas de dados previstas, ambientes virtuais e o
arquivo `.env`. Use somente vozes para as quais tenha autorização. O WAV enviado
pela interface é usado em uma pasta temporária local e removido após o cadastro;
os latentes de voz são dados pessoais e permanecem em `data/voices/`.

O código da biblioteca Coqui TTS e os pesos do XTTS-v2 têm licenças diferentes.
Os pesos do XTTS-v2 usam a [Coqui Public Model License](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt),
que permite apenas uso não comercial do modelo e de suas saídas. Revise seus
termos antes de baixar ou utilizar o modelo.

## Serviço de voz

O serviço também oferece uma API Python local. Depois de aceitar
os termos da licença do modelo, defina `COQUI_TOS_AGREED=1` no ambiente para o
primeiro uso. Caso os pesos não estejam no cache local, o Coqui poderá baixá-los.

```python
from pathlib import Path
from services.voice import get_or_create_voice_latents, list_available_voices, synthesize_speech

# Forneça uma lista com um ou mais WAVs autorizados.
get_or_create_voice_latents("minha_voz", [Path("data/voices/referencia.wav")])
print(list_available_voices())
metricas = synthesize_speech("Olá, esta é uma amostra.", "minha_voz", Path("data/podcasts/amostra.wav"))
print(metricas)
```

Crie o diretório de saída antes da síntese. Arquivos existentes não são
sobrescritos. O perfil contém os latentes em `data/voices/minha_voz.latents.pt`;
referências alteradas são detectadas quando fornecidas novamente à função de
criação. A listagem mostra perfis com latentes, não WAVs avulsos. A inferência é
local após o download do modelo.

## Serviço de áudio

Para montar um episódio a partir de um perfil já criado:

```python
from pathlib import Path
from services.audio import generate_podcast

resultado = generate_podcast(
    "Primeiro parágrafo do roteiro.\nSegundo parágrafo do roteiro.",
    "minha_voz",
    Path("data/podcasts/episodio.mp3"),
    progress_callback=lambda atual, total: print(f"Trecho {atual}/{total}"),
)
print(resultado)
```

O serviço divide o roteiro em trechos, cria pausas entre blocos e parágrafos e
usa o FFmpeg local para exportar MP3. Quando o caminho é omitido, um nome único
é criado em `data/podcasts/`. A saída existente nunca é sobrescrita. Arquivos
intermediários ficam em uma pasta temporária privada, removida ao terminar. A
duração e o RTF retornados são calculados antes da codificação, com as pausas.
Trechos com conteúdo incomum podem exceder o limite de tokens do XTTS mesmo com
até 220 caracteres; nesse caso, é necessário dividir o trecho manualmente.

Para executar os testes rápidos:

```bash
.venv/bin/python tests/test_voice_service.py
.venv/bin/python tests/test_audio_service.py
.venv/bin/python tests/test_app.py
```

O teste real opcional usa o modelo já baixado, as referências autorizadas locais
`13seg.wav`, `2.wav`, `3.wav` e executa duas sínteses em CPU:

```bash
.venv/bin/python tests/test_voice_service.py --integration --agree-license
.venv/bin/python tests/test_audio_service.py --integration --agree-license
```
