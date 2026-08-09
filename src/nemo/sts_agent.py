"""Nemo Slay the Spire agent using spirecomm and a two-tier LLM agent stack."""

import logging
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from spirecomm.ai.agent import SimpleAgent
from spirecomm.communication.action import (
    EndTurnAction,
    PlayCardAction,
    ProceedAction,
    StateAction,
)
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass

from nemo.agents import build_leader_agent, build_player_agent, get_llm
from nemo.tools import GameContext

LOG_FILE = Path(__file__).resolve().parents[2] / "nemo_sts.log"

# Keep the player's session history bounded so input tokens stay manageable.
MAX_SESSION_MESSAGES = 20

logger = logging.getLogger("nemo.sts")


def describe_action(action) -> str:
    """Human-readable version of the action being sent to the game."""
    if isinstance(action, PlayCardAction):
        target = getattr(action, "target_monster", None)
        tname = getattr(target, "name", None)
        if tname is None:
            tname = getattr(action, "target_index", None)
        return f"play hand[{action.card_index}] -> target {tname}"
    from spirecomm.communication.action import ChooseAction

    if isinstance(action, ChooseAction):
        return f"choose {action.choice_index or action.name}"
    return repr(action)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )


class NemoSpireAgent(SimpleAgent):
    """An agent that plays the game: the leader hands the whole game to the player agent."""

    def __init__(self, chosen_class=PlayerClass.IRONCLAD):
        super().__init__(chosen_class)
        llm = get_llm()
        self.context = GameContext()
        self.player = build_player_agent(llm, self.context)
        self.leader = build_leader_agent(llm, self.player, self.context)

    def handle_error(self, error):
        """The game rejected a command. Never raise here: request a fresh state and keep playing."""
        logger.warning("Game rejected a command: %s", error)
        return StateAction()

    def get_next_action_in_game(self, game_state):
        """Called whenever a new game state arrives."""
        if not game_state.play_available and not game_state.proceed_available:
            if game_state.end_available:
                return EndTurnAction()
            action = super().get_next_action_in_game(game_state)
            if action is not None:
                return action
            return StateAction()
        self.context.game_state = game_state
        self.context.action = None
        self.context.turn += 1
        try:
            if not self.context.delegated:
                self.leader.invoke(
                    {"messages": [("user", "Take over this entire game and hand it to the player agent.")]},
                    config={"recursion_limit": 4},
                )
                logger.info("Leader delegated the entire game to the player agent")
                self.context.messages = [
                    HumanMessage(
                        content="You are now playing Slay the Spire. Take the best action each turn "
                        "using take_action. Your previous actions are listed below for memory."
                    )
                ]
            self.context.messages.append(
                HumanMessage(
                    content=f"Turn {self.context.turn}: the game state just changed. "
                    "Record exactly one action."
                )
            )
            self.context.messages = self.context.messages[-MAX_SESSION_MESSAGES:]
            self.player.invoke(
                {"messages": self.context.messages},
                config={"recursion_limit": 6},
            )
            if self.context.action is not None:
                action = self.context.action
                self.context.messages.append(
                    HumanMessage(
                        content=f"Turn {self.context.turn} action taken: {describe_action(action)}."
                    )
                )
                self.context.messages = self.context.messages[-MAX_SESSION_MESSAGES:]
                logger.info("LLM action: %s", describe_action(action))
                return action
            logger.warning("No action recorded, defaulting to proceed")
            return ProceedAction()
        except Exception:
            logger.exception("LLM run failed, using recorded action or fallback")
            if self.context.action is not None:
                logger.info("LLM action (from interrupted run): %s", describe_action(self.context.action))
                return self.context.action
            fallback = super().get_next_action_in_game(game_state)
            if fallback is not None:
                return fallback
            return StateAction()


class GameRunState(TypedDict, total=False):
    result: str


def _game_controller(state: GameRunState) -> dict:
    """Run one whole game inside a single LangGraph node.

    The per-turn leader/player invocations happen inside this node's
    execution, so LangSmith records exactly one LangGraph entry per game
    with each turn's LLM calls as nested child runs.
    """
    agent = NemoSpireAgent()
    coordinator = Coordinator()
    coordinator.signal_ready()  # Sends "ready\n" to CommunicationMod
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    logger.info("Agent started, waiting for game...")
    coordinator.play_one_game(agent.chosen_class)
    return {"result": "done"}


def build_game_graph():
    """Compile the single-LangGraph-entry game controller."""
    graph = StateGraph(GameRunState)
    graph.add_node("game_controller", _game_controller)
    graph.add_edge(START, "game_controller")
    graph.add_edge("game_controller", END)
    return graph.compile()


def main() -> None:
    load_dotenv()
    setup_logging()
    game = build_game_graph()
    game.invoke({"result": ""}, config={"recursion_limit": 4})


if __name__ == "__main__":
    main()
