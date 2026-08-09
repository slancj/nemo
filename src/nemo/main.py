"""Nemo package."""

import os
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich import print

from nemo.agents import build_leader_agent, build_player_agent, get_llm
from nemo.tools import GameContext


class NemoState(TypedDict):
    message: NotRequired[str]
    response: NotRequired[str]


def leader_node(state: NemoState) -> NemoState:
    llm = get_llm()
    context = GameContext()
    leader = build_leader_agent(llm, build_player_agent(llm, context), context)
    reply = leader.invoke({"messages": [("user", state["message"])]})
    return {"response": str(reply["messages"][-1].content)}


def build_graph() -> StateGraph:
    graph = StateGraph(NemoState)
    graph.add_node("leader", leader_node)
    graph.add_edge(START, "leader")
    graph.add_edge("leader", END)
    return graph


def main() -> None:
    load_dotenv()
    result = build_graph().compile().invoke({"message": "Hello from Nemo!"})
    print(f"[bold blue]Nemo[/bold blue] ({os.getenv('NEMO_ENV', 'production')})")
    print(result["response"])


if __name__ == "__main__":
    main()
