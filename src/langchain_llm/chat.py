"""Prompt composition and bounded, in-memory conversation state."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable


class ChatSession:
    def __init__(self, model: Runnable, *, max_turns: int = 10):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self.history: list[HumanMessage | AIMessage] = []
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful assistant. Be clear, accurate, and concise. "
                        "Say when you are uncertain."
                    ),
                ),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        self.chain = prompt | model | StrOutputParser()

    def ask(self, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        # Commit history only after a successful model response.
        answer = self.chain.invoke({"history": list(self.history), "question": question})
        self.history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        self.history = self.history[-2 * self.max_turns :]
        return answer

    def reset(self) -> None:
        self.history.clear()
