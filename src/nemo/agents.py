"""Nemo agents: a leader (manager) agent and a ReAct player agent with STS tools."""

import logging
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from nemo.tools import GameContext, make_sts_tools

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
        "turn. Each turn, call take_action exactly once with the single best action. "
        "On reward/event screens choose a valuable option with 'choose' (skip is "
        "allowed but usually worse). In combat, play cards and end the turn. Never "
        "act a second time in a turn, never explain or show any thinking. Act "
        "immediately.",
        is_terminal=lambda: context.action is not None,
        observe=observe,
    )


def build_leader_agent(model: BaseChatModel, player_agent, context: GameContext):
    """Tier-1 manager agent that hands the entire game to the player agent."""

    @tool
    def delegate_to_player(instruction: str) -> str:
        """Take over the entire game: seed the player agent's session. Called once per game."""
        context.messages = [HumanMessage(content=instruction)]
        context.delegated = True
        return "Player agent is now in control of the entire game."

    return _build_react_graph(
        model,
        [delegate_to_player],
        "You are the leader of the Nemo Slay the Spire team. Your ONLY job is to "
        "hand the entire game to the player agent by calling delegate_to_player "
        "with the user's request. Do not analyze the game, do not explain, do not "
        "show any thinking, do not write any other text. Call the tool and stop.",
        is_terminal=lambda: context.delegated,
    )
