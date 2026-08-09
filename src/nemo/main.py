"""Nemo package."""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich import print

from nemo.agents import build_player_agent, get_llm
from nemo.tools import GameContext

_llm = get_llm()
_context = GameContext()


def main() -> None:
    load_dotenv()
    _context.messages = [
        HumanMessage(
            content="You are now playing Slay the Spire. Record your actions with take_action."
        )
    ]
    build_player_agent(_llm, _context).invoke(
        {"messages": _context.messages},
        config={"recursion_limit": 16},
    )
    print(f"[bold blue]Nemo[/bold blue] ({os.getenv('NEMO_ENV', 'production')})")
    print(f"actions: {[type(a).__name__ for a in _context.pending_actions]}")


if __name__ == "__main__":
    main()
