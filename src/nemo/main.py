"""Nemo package."""

import os
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich import print

from nemo.agents import build_leader_agent, build_player_agent, get_llm
from nemo.tools import GameContext

_llm = get_llm()
_context = GameContext()
_player = build_player_agent(_llm, _context)
_leader = build_leader_agent(_llm, _player, _context)


class NemoState(TypedDict):
    message: NotRequired[str]
    response: NotRequired[str]


def leader_node(state: NemoState) -> NemoState:
    _leader.invoke(
        {"messages": [("user", state["message"])]},
        config={"recursion_limit": 4},
    )
    _player.invoke(
        {"messages": _context.messages},
        config={"recursion_limit": 6},
    )
    return {"response": f"action: {_context.action.command if _context.action else None}"}


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
