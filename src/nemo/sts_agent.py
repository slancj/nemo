"""Nemo Slay the Spire agent using spirecomm and a two-tier LLM agent stack."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from spirecomm.ai.agent import SimpleAgent
from spirecomm.communication.action import EndTurnAction, ProceedAction
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass

from nemo.agents import build_leader_agent, build_player_agent, get_llm
from nemo.tools import GameContext

LOG_FILE = Path(__file__).resolve().parents[2] / "nemo_sts.log"

# Keep the player's session history bounded so input tokens stay manageable.
MAX_SESSION_MESSAGES = 20

logger = logging.getLogger("nemo.sts")


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

    def get_next_action_in_game(self, game_state):
        """Called whenever a new game state arrives."""
        if not game_state.play_available and not game_state.proceed_available:
            if game_state.end_available:
                return EndTurnAction()
            return super().get_next_action_in_game(game_state)
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
            self.context.messages.append(
                HumanMessage(
                    content=f"Turn {self.context.turn}: the game state just changed. "
                    "Record exactly one action."
                )
            )
            self.context.messages = self.context.messages[-MAX_SESSION_MESSAGES:]
            result = self.player.invoke(
                {"messages": self.context.messages},
                config={"recursion_limit": 6},
            )
            self.context.messages = result["messages"][-MAX_SESSION_MESSAGES:]
            if self.context.action is not None:
                logger.info("LLM action: %s", self.context.action)
                return self.context.action
            logger.warning("No action recorded, defaulting to proceed")
            return ProceedAction()
        except Exception:
            logger.exception("LLM run failed, using recorded action or fallback")
            if self.context.action is not None:
                logger.info("LLM action (from interrupted run): %s", self.context.action)
                return self.context.action
            return super().get_next_action_in_game(game_state)


def main() -> None:
    load_dotenv()
    setup_logging()
    agent = NemoSpireAgent()
    coordinator = Coordinator()
    coordinator.signal_ready()  # Sends "ready\n" to CommunicationMod
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    logger.info("Agent started, waiting for game...")
    coordinator.play_one_game(agent.chosen_class)


if __name__ == "__main__":
    main()
