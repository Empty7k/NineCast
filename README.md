# NineCast

Aplicativo pessoal para transformar conteúdos de estudo em podcasts narrados com
voz autorizada. O projeto está em desenvolvimento incremental; atualmente é
possível escrever e revisar um roteiro na interface. A geração de áudio ainda não
foi implementada e o texto não é salvo automaticamente.

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

Abra o endereço local mostrado no terminal. O texto escrito permanece na sessão
da página; copie-o antes de atualizá-la ou fechá-la. Para encerrar o servidor,
pressione `Ctrl+C` no terminal.

## Privacidade e dados

Não inclua gravações, documentos, roteiros pessoais, podcasts ou credenciais no
Git. O `.gitignore` exclui as pastas de dados previstas, ambientes virtuais e o
arquivo `.env`. Use somente vozes para as quais tenha autorização. A interface
atual não recebe gravações nem gera áudio.

O código da biblioteca Coqui TTS e os pesos do XTTS-v2 têm licenças diferentes.
Os pesos do XTTS-v2 usam a [Coqui Public Model License](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt),
que permite apenas uso não comercial do modelo e de suas saídas. Revise seus
termos antes de baixar ou utilizar o modelo.

## Próxima etapa

Validar o carregamento do XTTS-v2 em CPU e gerar uma amostra curta usando um WAV
de referência autorizado, medindo tempo e consumo de memória. O primeiro uso do
modelo exigirá download dos pesos e aceitação dos respectivos termos.
