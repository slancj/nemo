"""Slay the Spire tools for the player agent."""

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from spirecomm.communication.action import (
    CancelAction,
    ChooseAction,
    EndTurnAction,
    PlayCardAction,
    ProceedAction,
)


@dataclass
class GameContext:
    game_state: Any = None
    action: Any = None


def serialize_state(game_state) -> str:
    """Render a spirecomm game state for the LLM."""
    if game_state is None:
        return "No game state available yet."
    player = game_state.player
    hand = [
        f"{i + 1}. {c.name} (cost {c.cost}, playable={c.is_playable}, target={c.has_target})"
        for i, c in enumerate(game_state.hand)
    ]
    monsters = [
        f"{i + 1}. {m.name} hp={m.current_hp}/{m.max_hp} intent={m.intent}"
        for i, m in enumerate(game_state.monsters)
        if m.current_hp > 0
    ]
    available = [
        name
        for name, flag in [
            ("play", game_state.play_available),
            ("proceed", game_state.proceed_available),
            ("end", game_state.end_available),
            ("choose", game_state.choice_available),
            ("cancel", game_state.cancel_available),
        ]
        if flag
    ]
    return f"""Screen: {game_state.screen_type}
Player: hp={player.current_hp}/{player.max_hp} block={player.block} energy={player.energy}
Available commands: {available}
Hand:
{chr(10).join(hand) if hand else '(empty)'}
Monsters:
{chr(10).join(monsters) if monsters else '(none)'}
Room: {game_state.room_type}"""


class TakeAction(BaseModel):
    """The single next action to take in Slay the Spire."""

    command: Literal["play", "end", "proceed", "choose", "cancel"] = Field(
        description="What to do: play a card from the hand, end the turn, proceed to the next screen, choose an option, or cancel."
    )
    card_index: int | None = Field(
        default=None,
        description="1-based index of the card to play from the hand. Only set when command is 'play'.",
    )
    target_index: int | None = Field(
        default=None,
        description="Monster index to target with the card. Only set when command is 'play' and the card is an attack or otherwise needs a target.",
    )
    option_index: int | None = Field(
        default=None,
        description="0-based index of the option to choose. Only set when command is 'choose'.",
    )


def build_action(spec: TakeAction, game_state) -> Any:
    """Map a TakeAction spec onto a spirecomm action."""
    if spec.command == "play":
        if spec.card_index is None or not (1 <= spec.card_index <= len(game_state.hand)):
            return ProceedAction()
        target = None
        if spec.target_index is not None:
            for m in game_state.monsters:
                if m.current_hp > 0 and m.monster_index == spec.target_index:
                    target = m
                    break
        return PlayCardAction(card_index=spec.card_index - 1, target_monster=target)
    if spec.command == "end":
        return EndTurnAction()
    if spec.command == "proceed":
        return ProceedAction()
    if spec.command == "choose":
        if spec.option_index is None:
            return ProceedAction()
        return ChooseAction(spec.option_index)
    if spec.command == "cancel":
        return CancelAction()
    return ProceedAction()


def make_sts_tools(context: GameContext):
    """Build the tools the player agent uses to interact with the game."""

    @tool
    def get_game_state() -> str:
        """Return the current Slay the Spire game state: screen, player HP/energy, hand, monsters, and available commands."""
        return serialize_state(context.game_state)

    @tool(args_schema=TakeAction)
    def take_action(command, card_index=None, target_index=None, option_index=None) -> str:
        """Record the single next action to take in Slay the Spire. Hand indices are 1-based; monster and option indices are 0-based. Only set the fields relevant to your command."""
        if context.game_state is None:
            return "No game state available; no action taken."
        spec = TakeAction(
            command=command,
            card_index=card_index,
            target_index=target_index,
            option_index=option_index,
        )
        context.action = build_action(spec, context.game_state)
        return f"Action '{spec.command}' recorded."

    return [get_game_state, take_action]
