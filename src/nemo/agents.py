"""Nemo agents: a ReAct player agent with STS tools (single-agent, no leader)."""

import logging
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from nemo.tools import (
    MAX_ACTIONS_PER_TURN,
    TERMINAL_COMMANDS,
    GameContext,
    make_sts_tools,
)

logger = logging.getLogger("nemo.agents")


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


def _build_react_graph(
    model: BaseChatModel,
    tools: list,
    system_prompt: str,
    is_terminal,
    observe=None,
):
    """A ReAct-style loop that always stops right after a terminal tool runs.

    `observe` (optional) is called before every model invocation and prepends the
    result, so the game state is refreshed automatically each turn without an
    extra LLM round trip. `is_terminal()` is checked after every tool batch so we
    END immediately once the action is recorded instead of asking the proxy model
    to produce a final text answer (it never terminates an open-ended loop itself).
    """
    tool_node = ToolNode(tools)
    model_with_tools = model.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        observation = f"Current game state:\n{observe()}" if observe is not None else None
        messages = [
            SystemMessage(content=system_prompt),
            *([HumanMessage(content=observation)] if observation else []),
            *state["messages"],
        ]
        response = model_with_tools.invoke(messages)
        if isinstance(response, AIMessage) and response.tool_calls:
            response = AIMessage(
                content="",
                tool_calls=response.tool_calls,
                id=response.id,
                name=response.name,
            )
        return {"messages": [response]}

    def after_agent(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def after_tools(state: MessagesState) -> str:
        return END if is_terminal() else "agent"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    return graph.compile()


def build_player_agent(model: BaseChatModel, context: GameContext):
    """Tier-2 ReAct agent that plays Slay the Spire using the STS tools."""
    get_state_tool, take_action_tool = make_sts_tools(context)

    def observe() -> str:
        return get_state_tool.invoke({})

    return _build_react_graph(
        model,
        [take_action_tool],
        "You are an expert Slay the Spire player driving this entire game. The "
        "current game state is fetched for you automatically at the start of every "
        "turn, and is very detailed. All indices are 0-based and match the numbers "
        "shown in the game state.\n\n"
        "You may chain multiple actions in one turn: call take_action once per "
        "action, in order. In combat, play your strongest attacks (target the "
        "lowest-HP enemy), use block/defense when you are about to take heavy "
        "damage, then finish with take_action(command='end'). Do not end the turn "
        "while you still have playable cards and energy.\n\n"
        "- CARD_REWARD / COMBAT_REWARD / BOSS_REWARD / EVENT / REST / SHOP: ALWAYS "
        "take the best reward with take_action(command='choose', option_index=<the "
        "number shown>). Skipping rewards is usually a mistake. For a card reward, "
        "choose the card that best improves your deck.\n"
        "- MAP: move forward by choosing a reachable node with 'choose'.\n"
        "- Use 'proceed' to advance past non-choice screens.\n\n"
        "Examples:\n"
        "State shows '0. Strike ... playable=True' and '0. Cultist hp=20' -> "
        "take_action(command='play', card_index=0, target_index=0)\n"
        "State shows 'CARD_REWARD: 0. Iron Wave' -> "
        "take_action(command='choose', option_index=0)\n"
        "No useful card left -> take_action(command='end')\n\n"
        "Never explain, never show any thinking. Act immediately and stop once "
        "your turn's actions are recorded.",
        is_terminal=lambda: (
            context.last_command in TERMINAL_COMMANDS
            or len(context.pending_actions) >= MAX_ACTIONS_PER_TURN
        ),
        observe=observe,
    )
