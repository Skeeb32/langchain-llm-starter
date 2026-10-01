"""Terminal entry point. Run --demo to try the pipeline without a model server."""

import argparse
import os
import sys

from httpx import HTTPError
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from ollama import ResponseError

from langchain_llm.chat import ChatSession


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat using LangChain and Ollama")
    parser.add_argument("--prompt", help="Ask one question and exit")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2"))
    parser.add_argument(
        "--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument(
        "--demo", action="store_true", help="Use fixed responses; no real inference"
    )
    args = parser.parse_args(argv)
    if args.max_turns < 1:
        parser.error("--max-turns must be at least 1")
    if args.demo:
        model = FakeListChatModel(
            responses=[
                "Demo response: the LangChain pipeline works. Connect Ollama for real answers."
            ]
        )
    else:
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=args.model, base_url=args.base_url, client_kwargs={"timeout": 120.0}
        )
    session = ChatSession(model, max_turns=args.max_turns)
    try:
        if args.prompt is not None:
            print(session.ask(args.prompt))
            return 0
        print("LangChain chat — /reset clears history, /exit quits.")
        while True:
            question = input("You: ").strip()
            if question == "/exit":
                return 0
            if question == "/reset":
                session.reset()
                print("History cleared.")
            elif question:
                print(f"Assistant: {session.ask(question)}")
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    except (HTTPError, ResponseError, ConnectionError, ValueError) as exc:
        print(
            f"Request failed ({type(exc).__name__}). Check that Ollama is running, "
            "the model is downloaded, and the prompt is nonempty.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
