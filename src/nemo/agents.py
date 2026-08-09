"""Nemo agents: a leader (manager) agent and a ReAct player agent with STS tools."""

import logging
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
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
):
    """A ReAct-style loop that always stops right after a terminal tool runs.

    The proxy model never terminates an open-ended ReAct loop on its own, so we
    route END immediately once `is_terminal()` becomes true instead of asking the
    model to produce a final text answer.
    """
    tool_node = ToolNode(tools)
    model_with_tools = model.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        response = model_with_tools.invoke(
            [SystemMessage(content=system_prompt), *state["messages"]]
        )
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
    return _build_react_graph(
        model,
        make_sts_tools(context),
        "You are an expert Slay the Spire player. Follow this exact procedure:\n"
        "1. Call get_game_state to see the current state.\n"
        "2. Call take_action with exactly ONE action.\n"
        "Do not analyze, explain, justify, or show any thinking. "
        "Do not write any text before or after your tool calls. Act immediately.",
        is_terminal=lambda: context.action is not None,
    )


def build_leader_agent(model: BaseChatModel, player_agent, context: GameContext):
    """Tier-1 manager agent that delegates decisions to the player agent."""

    @tool
    def delegate_to_player(instruction: str) -> str:
        """Delegate deciding the current move to the expert Slay the Spire player agent."""
        try:
            player_agent.invoke(
                {"messages": [("user", instruction)]},
                config={"recursion_limit": 6},
            )
        except Exception:
            logger.exception("Player agent run did not terminate cleanly")
        if context.action is not None:
            return f"Player decided: {context.action.command}."
        return "Player did not record an action."

    return _build_react_graph(
        model,
        [delegate_to_player],
        "You are the leader of the Nemo Slay the Spire team. Your ONLY job is to "
        "call delegate_to_player with the user's request. Do not analyze the game, do not "
        "explain, do not show any thinking, do not write any other text. Call the tool and stop.",
        is_terminal=lambda: context.action is not None,
    )
