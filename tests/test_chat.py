import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from langchain_llm.chat import ChatSession
from langchain_llm.cli import main


def test_context_and_bounded_history():
    calls = []

    def respond(prompt):
        calls.append(prompt.to_messages())
        return AIMessage(content="answer")

    session = ChatSession(RunnableLambda(respond), max_turns=1)
    session.ask("first")
    session.ask("second")
    assert [m.content for m in calls[1]][1:] == ["first", "answer", "second"]
    assert [m.content for m in session.history] == ["second", "answer"]
    session.reset()
    assert session.history == []


def test_failure_does_not_change_history():
    def fail(_):
        raise RuntimeError("offline")

    session = ChatSession(RunnableLambda(fail))
    with pytest.raises(RuntimeError):
        session.ask("hello")
    assert session.history == []


def test_empty_prompt_never_calls_model():
    def unexpected(_):
        pytest.fail("model should not be called")

    session = ChatSession(RunnableLambda(unexpected))
    with pytest.raises(ValueError):
        session.ask("  ")


def test_invalid_history_limit():
    with pytest.raises(ValueError):
        ChatSession(RunnableLambda(lambda _: "answer"), max_turns=0)


def test_demo_cli(capsys):
    assert main(["--demo", "--prompt", "hello"]) == 0
    assert "Demo response" in capsys.readouterr().out


def test_cli_empty_prompt(capsys):
    assert main(["--demo", "--prompt", " "]) == 1
    assert "Request failed" in capsys.readouterr().err


def test_interactive_reset(monkeypatch, capsys):
    answers = iter(["hello", "/reset", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert main(["--demo"]) == 0
    assert "History cleared" in capsys.readouterr().out
