"""Slay the Spire tools for the player agent."""

from dataclasses import dataclass, field
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
from spirecomm.spire.screen import ScreenType


@dataclass
class GameContext:
    game_state: Any = None
    action: Any = None
    delegated: bool = False
    turn: int = 0
    messages: list = field(default_factory=list)


def _available(game_state) -> list:
    return [
        name
        for name, flag in [
            ("play", getattr(game_state, "play_available", False)),
            ("proceed", getattr(game_state, "proceed_available", False)),
            ("end", getattr(game_state, "end_available", False)),
            ("choose", getattr(game_state, "choice_available", False)),
            ("cancel", getattr(game_state, "cancel_available", False)),
        ]
        if flag
    ]


def _serialize_screen(game_state) -> str:
    """Render the screen's selectable options (rewards, events, shops, etc.)."""
    screen = getattr(game_state, "screen", None)
    st = getattr(screen, "screen_type", None) or getattr(game_state, "screen_type", None)
    if screen is None or st is None:
        return ""
    lines = []
    if st == ScreenType.CARD_REWARD:
        lines.append("Card reward (choose <i> to add it to your deck, or proceed to skip):")
        for i, c in enumerate(getattr(screen, "cards", None) or []):
            lines.append(
                f"{i}. {c.name} ({c.type.name} {c.rarity.name}, cost {c.cost}, upgrades {getattr(c, 'upgrades', 0)})"
            )
    elif st == ScreenType.COMBAT_REWARD:
        lines.append("Combat reward (choose <i> to take it, or proceed to leave):")
        for i, r in enumerate(getattr(screen, "rewards", None) or []):
            desc = getattr(r, "reward_type", "?")
            gold = getattr(r, "gold", 0)
            relic = getattr(r, "relic", None)
            potion = getattr(r, "potion", None)
            if relic is not None:
                desc = f"{desc} ({relic.name})"
            elif potion is not None:
                desc = f"{desc} ({potion.name})"
            elif gold:
                desc = f"{desc} ({gold} gold)"
            lines.append(f"{i}. {desc}")
    elif st == ScreenType.BOSS_REWARD:
        lines.append("Boss relic (choose <i>):")
        for i, r in enumerate(getattr(screen, "relics", None) or []):
            lines.append(f"{i}. {r.name}")
    elif st == ScreenType.EVENT:
        lines.append("Event options (choose <i>):")
        for i, opt in enumerate(getattr(screen, "options", None) or []):
            idx = getattr(opt, "choice_index", None)
            label = getattr(opt, "label", None) or getattr(opt, "text", None) or ""
            disabled = getattr(opt, "disabled", False)
            lines.append(f"{idx if idx is not None else i}. {label}{' [disabled]' if disabled else ''}")
    elif st == ScreenType.REST:
        lines.append("Rest options (choose <i>):")
        for i, o in enumerate(getattr(screen, "rest_options", None) or []):
            lines.append(f"{i}. {o}")
    elif st == ScreenType.SHOP_SCREEN:
        lines.append("Shop (proceed to exit):")
        for i, c in enumerate(getattr(screen, "cards", None) or []):
            lines.append(f"card {i}: {c.name} cost {getattr(c, 'cost', 0)}")
        for i, r in enumerate(getattr(screen, "relics", None) or []):
            lines.append(f"relic {i}: {r.name}")
        for i, p in enumerate(getattr(screen, "potions", None) or []):
            lines.append(f"potion {i}: {getattr(p, 'name', '?')}")
    elif st == ScreenType.GRID:
        lines.append(f"Grid select (pick {getattr(screen, 'num_cards', '?')} card(s)):")
        for i, c in enumerate(getattr(screen, "cards", None) or []):
            lines.append(f"{i}. {c.name} cost {getattr(c, 'cost', 0)}")
    elif st == ScreenType.HAND_SELECT:
        lines.append(f"Hand select (pick {getattr(screen, 'num_cards', '?')} card(s)):")
        for i, c in enumerate(getattr(screen, "cards", None) or []):
            lines.append(f"{i}. {c.name} cost {getattr(c, 'cost', 0)}")
    elif st == ScreenType.MAP:
        lines.append("Map (choose <i> to move to that node):")
        for i, n in enumerate(getattr(screen, "next_nodes", None) or []):
            lines.append(f"{i}. node ({n.x},{n.y}) room {n.symbol}")
        if getattr(screen, "boss_available", False):
            lines.append("Boss node is reachable.")
    elif st == ScreenType.CHEST:
        lines.append(f"Chest: {getattr(screen, 'chest_type', '?')}")
    elif st == ScreenType.SHOP_ROOM:
        lines.append("Entering the shop (proceed to open it).")
    elif st == ScreenType.GAME_OVER:
        lines.append(
            f"Game over (score {getattr(screen, 'score', '?')}, victory={getattr(screen, 'victory', '?')})."
        )
    elif st == ScreenType.COMPLETE:
        lines.append("Victory! The run is complete.")
    return "\n".join(lines)


def serialize_state(game_state) -> str:
    """Render a spirecomm game state for the LLM. Safe for non-combat states."""
    if game_state is None:
        return "No game state available yet."
    screen = getattr(game_state, "screen_type", None)
    room = getattr(game_state, "room_type", None)
    available = _available(game_state)
    screen_details = _serialize_screen(game_state)
    player = getattr(game_state, "player", None)
    if player is None:
        return (
            f"Screen: {screen}\nRoom: {room}\nPlayer: (not in combat)\n"
            f"Available commands: {available}\n{screen_details or '(No combat state to inspect.)'}"
        )
    hand = getattr(game_state, "hand", None) or []
    monsters = getattr(game_state, "monsters", None) or []
    hand_lines = [
        f"{i}. {c.name} (cost {c.cost}, playable={c.is_playable}, target={c.has_target})"
        for i, c in enumerate(hand)
    ]
    monster_lines = [
        f"{m.monster_index}. {m.name} hp={m.current_hp}/{m.max_hp} intent={m.intent}"
        for m in monsters
        if m is not None and m.current_hp > 0
    ]
    return f"""Screen: {screen}
Player: hp={player.current_hp}/{player.max_hp} block={getattr(player, "block", 0)} energy={player.energy}
Available commands: {available}
Hand:
{chr(10).join(hand_lines) if hand_lines else '(empty)'}
Monsters:
{chr(10).join(monster_lines) if monster_lines else '(none)'}
Room: {room}
{screen_details}"""


class TakeAction(BaseModel):
    """The single next action to take in Slay the Spire."""

    command: Literal["play", "end", "proceed", "choose", "cancel"] = Field(
        description="What to do: play a card from the hand, end the turn, proceed to the next screen, choose an option, or cancel."
    )
    card_index: int | None = Field(
        default=None,
        description="0-based index of the card to play from the hand. Only set when command is 'play'.",
    )
    target_index: int | None = Field(
        default=None,
        description="0-based monster index to target with the card. Only set when command is 'play' and the card is an attack or otherwise needs a target.",
    )
    option_index: int | None = Field(
        default=None,
        description="0-based index of the option to choose. Only set when command is 'choose'.",
    )


def build_action(spec: TakeAction, game_state) -> Any:
    """Map a TakeAction spec onto a spirecomm action. Indices are all 0-based."""
    if game_state is None:
        return ProceedAction()
    if spec.command == "play":
        hand = getattr(game_state, "hand", None) or []
        player = getattr(game_state, "player", None)
        energy = getattr(player, "energy", 0) if player is not None else 0

        def playable(i: int) -> bool:
            return (
                0 <= i < len(hand)
                and getattr(hand[i], "is_playable", True)
                and getattr(hand[i], "cost", 0) <= energy
            )

        idx = spec.card_index if spec.card_index is not None and playable(spec.card_index) else None
        if idx is None:
            idx = next((j for j in range(len(hand)) if playable(j)), None)
        if idx is None:
            return ProceedAction()
        card = hand[idx]
        monsters = getattr(game_state, "monsters", None) or []
        target = None
        if spec.target_index is not None:
            for m in monsters:
                if m is not None and m.current_hp > 0 and m.monster_index == spec.target_index:
                    target = m
                    break
        if target is None and getattr(card, "has_target", False):
            for m in monsters:
                if m is not None and m.current_hp > 0:
                    target = m
                    break
        return PlayCardAction(card_index=idx, target_monster=target)
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
        """Record the single next action to take in Slay the Spire. Hand, monster, and option indices are all 0-based and match the numbers shown in the game state. Only set the fields relevant to your command."""
        if context.game_state is None:
            return "No game state available; no action taken."
        if context.action is not None:
            return "You already recorded an action for this turn; do not act again."
        spec = TakeAction(
            command=command,
            card_index=card_index,
            target_index=target_index,
            option_index=option_index,
        )
        context.action = build_action(spec, context.game_state)
        return f"Action '{spec.command}' recorded."

    return [get_game_state, take_action]
