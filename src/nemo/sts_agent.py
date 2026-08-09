"""Nemo Slay the Spire agent using spirecomm and a ReAct LLM agent."""

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

from nemo.agents import build_player_agent, get_llm
from nemo.tools import GameContext, revalidate_action

LOG_FILE = Path(__file__).resolve().parents[2] / "nemo_sts.log"

# Keep the player's session history bounded so input tokens stay manageable.
MAX_SESSION_MESSAGES = 20

logger = logging.getLogger("nemo.sts")


def describe_action(action) -> str:
    """Human-readable version of the action being sent to the game."""
    if isinstance(action, PlayCardAction):
        card = getattr(action, "card", None)
        target = getattr(action, "target_monster", None)
        tname = getattr(target, "name", None)
        if tname is None:
            tname = getattr(action, "target_index", None)
        if card is not None:
            return f"play {card.name} -> target {tname}"
        return f"play hand[{action.card_index}] -> target {tname}"
    from spirecomm.communication.action import (
        CancelAction,
        ChooseAction,
        EndTurnAction,
        ProceedAction,
        StateAction,
    )

    if isinstance(action, EndTurnAction):
        return "end turn"
    if isinstance(action, ProceedAction):
        return "proceed"
    if isinstance(action, CancelAction):
        return "cancel"
    if isinstance(action, StateAction):
        return "state"
    if isinstance(action, ChooseAction):
        return f"choose {action.choice_index or action.name}"
    return repr(action)


class SafeCoordinator(Coordinator):
    """A Coordinator that never lets a stale action crash the game loop.

    The game can move to a newer state between when an action is planned and
    when it is executed (e.g. the mod reports ready_for_command=False while an
    effect resolves, then a fresh state arrives). A queued PlayCardAction may
    then reference a card that is no longer in hand, making spirecomm's
    `hand.index(card)` raise. We catch that, drop the action, and ask for a
    fresh state instead of dying.
    """

    def execute_next_action_if_ready(self):
        if len(self.action_queue) > 0 and self.action_queue[0].can_be_executed(self):
            action = self.action_queue[0]
            try:
                self.execute_next_action()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Dropping stale action '%s', requesting fresh state", describe_action(action)
                )
                self.add_action_to_queue(StateAction())


class GameRunState(TypedDict, total=False):
    result: str


def _game_controller(state: GameRunState) -> dict:
    """Run one whole game inside a single LangGraph node.

    The per-turn player invocations happen inside this node's execution, so
    LangSmith records exactly one LangGraph entry per game with each turn's LLM
    calls as nested child runs.
    """
    agent = NemoSpireAgent()
    coordinator = SafeCoordinator()
    coordinator.signal_ready()  # Sends "ready\n" to CommunicationMod
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    logger.info("Agent started, waiting for game...")
    coordinator.play_one_game(agent.chosen_class)
    return {"result": "done"}


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
    """A single agent that plays Slay the Spire: one LLM decision per turn, chaining multiple plays."""

    def __init__(self, chosen_class=PlayerClass.IRONCLAD):
        super().__init__(chosen_class)
        llm = get_llm()
        self.context = GameContext()
        self.context.messages = [
            HumanMessage(
                content="You are now playing Slay the Spire. Each turn, record your actions with "
                "take_action: chain multiple plays in combat, then end the turn. Your past "
                "actions are listed below for memory."
            )
        ]
        self.player = build_player_agent(llm, self.context)

    def handle_error(self, error):
        """The game rejected a command. Never raise here: request a fresh state and keep playing."""
        logger.warning("Game rejected a command: %s", error)
        return StateAction()

    def _pop_valid_action(self, game_state):
        """Pop the next queued action, dropping stale chained plays that are no longer valid."""
        while self.context.pending_actions:
            action = self.context.pending_actions.pop(0)
            fresh = revalidate_action(action, game_state)
            if fresh is not None:
                return fresh
            logger.warning("Dropping stale action: %s", describe_action(action))
        return None

    def get_next_action_in_game(self, game_state):
        """Called whenever a new game state arrives."""
        self.context.game_state = game_state
        if self.context.pending_actions:
            action = self._pop_valid_action(game_state)
            if action is not None:
                return action
        if not game_state.play_available and not game_state.proceed_available:
            if game_state.end_available:
                return EndTurnAction()
            action = super().get_next_action_in_game(game_state)
            if action is not None:
                return action
            return StateAction()
        self.context.turn += 1
        try:
            self.context.messages.append(
                HumanMessage(
                    content=f"Turn {self.context.turn}: the game state just changed. "
                    "Decide this turn's actions now."
                )
            )
            self.context.messages = self.context.messages[-MAX_SESSION_MESSAGES:]
            self.player.invoke(
                {"messages": self.context.messages},
                config={"recursion_limit": 16},
            )
            if self.context.pending_actions:
                planned = "; ".join(describe_action(a) for a in self.context.pending_actions)
                action = self._pop_valid_action(game_state)
                if action is not None:
                    self.context.messages.append(
                        HumanMessage(content=f"Turn {self.context.turn} actions: {planned}.")
                    )
                    self.context.messages = self.context.messages[-MAX_SESSION_MESSAGES:]
                    logger.info("LLM actions: %s", planned)
                    return action
            logger.warning("No action recorded, defaulting to proceed")
            return ProceedAction()
        except Exception:
            logger.exception("LLM run failed, using recorded action or fallback")
            action = self._pop_valid_action(game_state)
            if action is not None:
                return action
            fallback = super().get_next_action_in_game(game_state)
            if fallback is not None:
                return fallback
            return StateAction()


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
