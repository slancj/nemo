"""Nemo Slay the Spire agent using spirecomm and a two-tier LLM agent stack."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from spirecomm.ai.agent import SimpleAgent
from spirecomm.communication.action import EndTurnAction, ProceedAction
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass

from nemo.agents import build_leader_agent, build_player_agent, get_llm
from nemo.tools import GameContext

LOG_FILE = Path(__file__).resolve().parents[2] / "nemo_sts.log"

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
    """An agent that uses a leader -> player (ReAct) stack with STS tools to decide each action."""

    def __init__(self, chosen_class=PlayerClass.IRONCLAD):
        super().__init__(chosen_class)
        llm = get_llm()
        self.context = GameContext()
        self.leader = build_leader_agent(
            llm, build_player_agent(llm, self.context), self.context
        )

    def get_next_action_in_game(self, game_state):
        """Called whenever a new game state arrives. Run the leader -> player agents."""
        if not game_state.play_available and not game_state.proceed_available:
            if game_state.end_available:
                return EndTurnAction()
            return super().get_next_action_in_game(game_state)
        try:
            self.context.game_state = game_state
            self.context.action = None
            self.leader.invoke(
                {"messages": [("user", "Decide the best action for the current Slay the Spire state.")]},
                config={"recursion_limit": 6},
            )
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
