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
    pending_actions: list = field(default_factory=list)
    last_command: str | None = None
    turn: int = 0
    messages: list = field(default_factory=list)


MAX_ACTIONS_PER_TURN = 5
TERMINAL_COMMANDS = ("end", "proceed", "cancel")


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


def _powers_str(powers) -> str:
    if not powers:
        return "none"
    return ", ".join(
        f"{getattr(p, 'power_name', getattr(p, 'name', '?'))}({getattr(p, 'amount', 0)})"
        for p in powers
    )


def _card_str(c) -> str:
    return (
        f"{c.name} type={c.type.name} rarity={c.rarity.name} cost={c.cost} "
        f"playable={c.is_playable} target={c.has_target} upgraded={bool(c.upgrades)} "
        f"exhausts={c.exhausts}"
    )


def _monster_str(m) -> str:
    intent = getattr(m, "intent", None)
    intent_str = intent.name if intent is not None else "?"
    dmg = ""
    if intent is not None and intent.name.startswith("ATTACK"):
        dmg = f" incoming~{getattr(m, 'move_adjusted_damage', 0)}dmg x{getattr(m, 'move_hits', 0)}"
    return (
        f"{m.monster_index}. {m.name} hp={m.current_hp}/{m.max_hp} block={m.block} "
        f"intent={intent_str}{dmg} powers={_powers_str(getattr(m, 'powers', None) or [])} "
        f"half_dead={m.half_dead} gone={m.is_gone}"
    )


def _pile_str(pile, label: str) -> str:
    if not pile:
        return f"{label}: empty"
    counts: dict = {}
    for c in pile:
        counts[c.name] = counts.get(c.name, 0) + 1
    return f"{label} ({len(pile)}): " + ", ".join(f"{n} x{cnt}" for n, cnt in sorted(counts.items()))


def _serialize_screen(game_state) -> str:
    """Render the screen's selectable options verbosely (rewards, events, shops, etc.)."""
    screen = getattr(game_state, "screen", None)
    st = getattr(screen, "screen_type", None) or getattr(game_state, "screen_type", None)
    if screen is None or st is None:
        return ""
    lines = []
    if st == ScreenType.CARD_REWARD:
        lines.append(
            "CARD_REWARD: choose the card you want with 'choose' + option_index (0-based). "
            "Use 'proceed' to skip. Rewards are usually worth taking."
        )
        lines.append(f"  can_skip={getattr(screen, 'can_skip', False)} can_bowl={getattr(screen, 'can_bowl', False)}")
        for i, c in enumerate(getattr(screen, "cards", None) or []):
            lines.append(f"  {i}. {_card_str(c)}")
    elif st == ScreenType.COMBAT_REWARD:
        lines.append(
            "COMBAT_REWARD: choose a reward with 'choose' + option_index, or 'proceed' to leave."
        )
        for i, r in enumerate(getattr(screen, "rewards", None) or []):
            desc = str(getattr(r, "reward_type", "?"))
            gold = getattr(r, "gold", 0)
            relic = getattr(r, "relic", None)
            potion = getattr(r, "potion", None)
            if relic is not None:
                desc = f"{desc} ({relic.name})"
            elif potion is not None:
                desc = f"{desc} ({potion.name})"
            elif gold:
                desc = f"{desc} ({gold} gold)"
            lines.append(f"  {i}. {desc}")
    elif st == ScreenType.BOSS_REWARD:
        lines.append("BOSS_REWARD: choose one boss relic with 'choose' + option_index.")
        for i, r in enumerate(getattr(screen, "relics", None) or []):
            lines.append(f"  {i}. {r.name}")
    elif st == ScreenType.EVENT:
        lines.append("EVENT: choose an option with 'choose' + option_index.")
        lines.append(f"  event={getattr(screen, 'event_id', '?')} name={getattr(screen, 'event_name', '?')}")
        body = getattr(screen, "body_text", "")
        if body:
            lines.append(f"  body: {body}")
        for i, opt in enumerate(getattr(screen, "options", None) or []):
            idx = getattr(opt, "choice_index", None)
            label = getattr(opt, "label", None) or getattr(opt, "text", None) or ""
            disabled = getattr(opt, "disabled", False)
            lines.append(f"  {idx if idx is not None else i}. {label}{' [disabled]' if disabled else ''}")
    elif st == ScreenType.REST:
        lines.append("REST: choose a rest option with 'choose' + option_index.")
        lines.append(f"  has_rested={getattr(screen, 'has_rested', False)}")
        for i, o in enumerate(getattr(screen, "rest_options", None) or []):
            lines.append(f"  {i}. {o}")
    elif st == ScreenType.SHOP_SCREEN:
        lines.append("SHOP_SCREEN: buy with 'choose' (by name or index), or 'proceed' to exit.")
        lines.append(f"  gold={getattr(game_state, 'gold', 0)}")
        if getattr(screen, "purge_available", False):
            lines.append(f"  card removal available for {getattr(screen, 'purge_cost', 0)} gold")
        for i, c in enumerate(getattr(screen, "cards", None) or []):
            lines.append(f"  card {i}: {c.name} cost {getattr(c, 'price', 0)}")
        for i, r in enumerate(getattr(screen, "relics", None) or []):
            lines.append(f"  relic {i}: {r.name} cost {getattr(r, 'price', '?')}")
        for i, p in enumerate(getattr(screen, "potions", None) or []):
            lines.append(f"  potion {i}: {getattr(p, 'name', '?')} cost {getattr(p, 'price', '?')}")
    elif st == ScreenType.GRID:
        lines.append(f"GRID_SELECT: pick {getattr(screen, 'num_cards', '?')} card(s) with 'choose'.")
        lines.append(
            f"  any_number={getattr(screen, 'any_number', False)} confirm_up={getattr(screen, 'confirm_up', False)} "
            f"for_upgrade={getattr(screen, 'for_upgrade', False)} for_transform={getattr(screen, 'for_transform', False)} "
            f"for_purge={getattr(screen, 'for_purge', False)}"
        )
        for i, c in enumerate(getattr(screen, "cards", None) or []):
            lines.append(f"  {i}. {_card_str(c)}")
        sel = getattr(screen, "selected_cards", None) or []
        if sel:
            lines.append(f"  selected so far: {[c.name for c in sel]}")
    elif st == ScreenType.HAND_SELECT:
        lines.append(f"HAND_SELECT: pick {getattr(screen, 'num_cards', '?')} card(s) with 'choose'.")
        for i, c in enumerate(getattr(screen, "cards", None) or []):
            lines.append(f"  {i}. {_card_str(c)}")
        sel = getattr(screen, "selected_cards", None) or []
        if sel:
            lines.append(f"  selected so far: {[c.name for c in sel]}")
    elif st == ScreenType.MAP:
        lines.append("MAP: choose a node to move forward with 'choose' + option_index.")
        cur = getattr(screen, "current_node", None)
        if cur is not None:
            lines.append(f"  current position: ({cur.x},{cur.y}) {cur.symbol}")
        for i, n in enumerate(getattr(screen, "next_nodes", None) or []):
            lines.append(f"  {i}. node ({n.x},{n.y}) room {n.symbol}")
        if getattr(screen, "boss_available", False):
            lines.append("  boss node is reachable ('choose' the boss when available)")
    elif st == ScreenType.CHEST:
        lines.append(
            f"CHEST ({getattr(screen, 'chest_type', '?')}): use 'choose' to open it."
        )
        if getattr(screen, "chest_open", False):
            lines.append("  chest is open.")
    elif st == ScreenType.SHOP_ROOM:
        lines.append("SHOP_ROOM: entering the shop; use 'proceed' to open it.")
    elif st == ScreenType.GAME_OVER:
        lines.append(
            f"GAME_OVER (score {getattr(screen, 'score', '?')}, victory={getattr(screen, 'victory', '?')})."
        )
    elif st == ScreenType.COMPLETE:
        lines.append("COMPLETE: victory! The run is complete.")
    return "\n".join(lines)


def serialize_state(game_state) -> str:
    """Render a spirecomm game state verbosely so the LLM has full information."""
    if game_state is None:
        return "No game state available yet."
    lines = []
    lines.append("=== GAME ===")
    lines.append(
        f"Screen: {getattr(game_state, 'screen_type', None)} | Room: {getattr(game_state, 'room_type', None)} "
        f"| Floor: {getattr(game_state, 'floor', '?')} | Act: {getattr(game_state, 'act', '?')} "
        f"| Gold: {getattr(game_state, 'gold', '?')} | Turn: {getattr(game_state, 'turn', '?')}"
    )
    lines.append(
        f"Character: {getattr(game_state, 'character', '?')} | Ascension: {getattr(game_state, 'ascension_level', '?')} "
        f"| Seed: {getattr(game_state, 'seed', '?')} | In combat: {getattr(game_state, 'in_combat', False)}"
    )
    lines.append(f"Available commands: {', '.join(_available(game_state)) or 'none'}")
    relics = getattr(game_state, "relics", None) or []
    lines.append("Relics: " + (", ".join(getattr(r, "name", "?") for r in relics) if relics else "none"))
    deck = getattr(game_state, "deck", None) or []
    if deck:
        counts: dict = {}
        for c in deck:
            counts[c.name] = counts.get(c.name, 0) + 1
        lines.append(
            f"Deck ({len(deck)}): " + ", ".join(f"{n} x{cnt}" for n, cnt in sorted(counts.items()))
        )
    else:
        lines.append("Deck: empty")
    potions = getattr(game_state, "potions", None) or []
    if potions:
        lines.append(
            "Potions: "
            + ", ".join(
                f"slot {i}: {p.name} (use={getattr(p, 'can_use', '?')}, discard={getattr(p, 'can_discard', '?')}, target={getattr(p, 'requires_target', '?')})"
                for i, p in enumerate(potions)
            )
        )
    else:
        lines.append("Potions: none")

    player = getattr(game_state, "player", None)
    if player is not None:
        lines.append("")
        lines.append("=== COMBAT STATE ===")
        lines.append(
            f"Player: hp={player.current_hp}/{player.max_hp} block={getattr(player, 'block', 0)} "
            f"energy={player.energy} | Player powers: {_powers_str(getattr(player, 'powers', None) or [])}"
        )
        orbs = getattr(player, "orbs", None) or []
        if orbs:
            lines.append("Orbs: " + ", ".join(str(o) for o in orbs))
        hand = getattr(game_state, "hand", None) or []
        lines.append("Hand:")
        lines.append("\n".join(f"  {i}. {_card_str(c)}" for i, c in enumerate(hand)) if hand else "  (empty)")
        monsters = [m for m in getattr(game_state, "monsters", None) or [] if m is not None]
        lines.append("Monsters:")
        lines.append("\n".join(f"  {_monster_str(m)}" for m in monsters) if monsters else "  (none)")
        lines.append(_pile_str(getattr(game_state, "draw_pile", None) or [], "Draw pile"))
        lines.append(_pile_str(getattr(game_state, "discard_pile", None) or [], "Discard pile"))
        lines.append(_pile_str(getattr(game_state, "exhaust_pile", None) or [], "Exhaust pile"))
        limbo = getattr(game_state, "limbo", None) or []
        if limbo:
            lines.append("Limbo: " + ", ".join(c.name for c in limbo))
        cip = getattr(game_state, "card_in_play", None)
        if cip is not None:
            lines.append(f"Card in play: {cip.name}")
        lines.append(f"Cards discarded this turn: {getattr(game_state, 'cards_discarded_this_turn', 0)}")
    else:
        lines.append("")
        lines.append("(Not in combat.)")

    screen_details = _serialize_screen(game_state)
    if screen_details:
        lines.append("")
        lines.append("=== SCREEN ===")
        lines.append(screen_details)
    choice_list = getattr(game_state, "choice_list", None)
    if choice_list:
        lines.append("")
        lines.append(f"Choices (choose index -> value): {list(choice_list)}")

    dungeon_map = getattr(game_state, "map", None)
    nodes = getattr(dungeon_map, "nodes", None) if dungeon_map is not None else None
    if nodes:
        lines.append("")
        lines.append("=== MAP ===")
        for y in sorted(nodes.keys()):
            row = nodes[y]
            lines.append(
                "  floor " + str(y) + ": " + " ".join(f"({x},{y}){row[x].symbol}" for x in sorted(row.keys()))
            )
    return "\n".join(lines)


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
        return PlayCardAction(card=card, target_monster=target)
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


def revalidate_action(action, game_state) -> Any:
    """Return a fresh version of a queued action valid for the current state.

    Stale chained actions are dropped (return None) or re-targeted so we never
    send a play for a card that is no longer in hand / affordable.
    """
    if isinstance(action, PlayCardAction) and action.card is not None:
        hand = getattr(game_state, "hand", None) or []
        monsters = getattr(game_state, "monsters", None) or []
        player = getattr(game_state, "player", None)
        energy = getattr(player, "energy", 0) if player is not None else 0
        card = next((c for c in hand if c.uuid == action.card.uuid), None)
        if card is None or not getattr(card, "is_playable", False) or getattr(card, "cost", 0) > energy:
            return None
        target = None
        if getattr(card, "has_target", False):
            idx = getattr(getattr(action, "target_monster", None), "monster_index", None)
            target = next(
                (m for m in monsters if m is not None and m.current_hp > 0 and m.monster_index == idx),
                None,
            )
            if target is None:
                target = next((m for m in monsters if m is not None and m.current_hp > 0), None)
            if target is None:
                return None
        return PlayCardAction(card=card, target_monster=target)
    return action


def make_sts_tools(context: GameContext):
    """Build the tools the player agent uses to interact with the game."""

    @tool
    def get_game_state() -> str:
        """Return the current Slay the Spire game state: screen, player HP/energy, hand, monsters, and available commands."""
        return serialize_state(context.game_state)

    @tool(args_schema=TakeAction)
    def take_action(command, card_index=None, target_index=None, option_index=None) -> str:
        """Record the next action to take in Slay the Spire. Hand, monster, and option indices are all 0-based and match the numbers shown in the game state. You may call this several times in a row to chain multiple plays in one turn (e.g. play Strike then play another card), finishing with 'end' in combat or 'proceed'/'choose' on screens. Only set the fields relevant to your command."""
        if context.game_state is None:
            return "No game state available; no action taken."
        spec = TakeAction(
            command=command,
            card_index=card_index,
            target_index=target_index,
            option_index=option_index,
        )
        context.pending_actions.append(build_action(spec, context.game_state))
        context.last_command = spec.command
        n = len(context.pending_actions)
        hint = " End the turn with 'end' when done." if spec.command == "play" else ""
        return f"Action '{spec.command}' recorded (#{n} so far). You may chain more actions.{hint}"

    return [get_game_state, take_action]
