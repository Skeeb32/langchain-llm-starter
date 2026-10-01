# LangChain LLM starter

Build a conversational LLM application from an empty project using Python, LangChain, and a local Ollama model. This project implements the application layer; it does **not** train model weights from scratch.

## Quick start

Requires Python 3.11+.

```sh
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
langchain-llm --demo --prompt "Hello"
```

Demo mode uses deterministic canned responses to verify the pipeline without downloads, credentials, or a running model.

## Real model inference

Install [Ollama](https://ollama.com/download), start its service, and download a model:

```sh
ollama pull llama3.2
langchain-llm --prompt "Explain LangChain in three sentences"
langchain-llm
```

If the Ollama service is not running, start it with `ollama serve`. Model download size and memory requirements depend on the model you select.

Interactive commands: `/reset` clears history; `/exit` quits. Ctrl+C or EOF also exits.

```sh
langchain-llm --model llama3.2 --max-turns 5
langchain-llm --base-url http://localhost:11434
```

`OLLAMA_MODEL` and `OLLAMA_BASE_URL` provide environment defaults; command-line flags override them. No API key is needed for a local Ollama server. Prompts go to the configured server, so choose a trusted endpoint.

## How it works

`ChatPromptTemplate → chat model → StrOutputParser`

`ChatSession` composes a system instruction, recent conversation messages, and the new question. After a successful response it retains at most the configured number of completed turns in memory. History is never saved to disk. This limit counts turns, not tokens; large prompts can still exceed a model's context window. Requests time out after 120 seconds.

- `src/langchain_llm/chat.py`: reusable conversation pipeline
- `src/langchain_llm/cli.py`: interactive and single-question CLI
- `tests/test_chat.py`: offline tests for context, reset, failure handling, and CLI
- `.github/workflows/ci.yml`: lint and tests on Python 3.11–3.13

## Development

```sh
python -m pip install -e '.[dev]'
ruff check .
pytest -q
```

Tests and demo mode do not validate real model quality or Ollama connectivity. Run a real prompt after installing Ollama to verify your inference environment.

Dependencies use compatible major-version ranges; installations may resolve different minor versions. This is an educational starter, without a web server, document retrieval, or persistent conversation storage.

References: [LangChain models](https://docs.langchain.com/oss/python/langchain/models) and [Ollama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama).
