"""Nemo package."""

import os
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class NemoState(TypedDict):
    message: NotRequired[str]
    response: NotRequired[str]


def greet(state: NemoState) -> NemoState:
    return {"message": f"Welcome to Nemo! ({os.getenv('NEMO_ENV', 'production')})"}


def ask_llm(state: NemoState) -> NemoState:
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    reply = llm.invoke(state["message"])
    return {"response": str(reply.content)}


def build_graph() -> StateGraph:
    graph = StateGraph(NemoState)
    graph.add_node("greet", greet)
    graph.add_node("ask_llm", ask_llm)
    graph.add_edge(START, "greet")
    graph.add_edge("greet", "ask_llm")
    graph.add_edge("ask_llm", END)
    return graph


def main() -> None:
    load_dotenv()
    result = build_graph().compile().invoke({"message": "Hello from Nemo!"})
    print(result["message"])
    print(result["response"])


if __name__ == "__main__":
    main()
