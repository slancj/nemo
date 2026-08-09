# nemo

Slay the Spire AI agent built with LangGraph, LangChain, and spirecomm.

## Setup

```bash
uv sync
```

## Run the STS agent

CommunicationMod (stdin/stdout, no socket) launches the agent itself. After a reboot, restore the config:

```bash
mkdir -p ~/.config/ModTheSpire/CommunicationMod
cat > ~/.config/ModTheSpire/CommunicationMod/config.properties <<'EOF'
command=/home/trueking/Safe/Proj/nemo/.venv/bin/python -m nemo.sts_agent
runAtGameStart=true
verbose=true
maxInitializationTimeout=20
EOF
```

Then start the game:

```bash
cd "/home/trueking/Safe/Files/Slay the Spire v2.3.4" && ./launch.sh
```

The mod spawns `nemo.sts_agent` automatically and the agent plays the game.

### Logs

stdout is the CommunicationMod protocol channel, so the agent must not `print()` anything visible. Logs go to:

- `nemo_sts.log` (in the project root) — everything: game states, LLM replies, decisions
- `communication_mod_errors.log` (in the game folder) — agent stderr/tracebacks

Watch live:

```bash
tail -f nemo_sts.log
```

### How the AI decides

`sts_agent.py` serializes each game state (screen, HP, energy, hand, monsters) into a prompt, calls the LLM from `.env` (`OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`), and the reply is parsed into a spirecomm action (`play N M`, `end`, `proceed`, `choose N`, `cancel`). If the LLM call fails, it falls back to `SimpleAgent`'s built-in logic.

## Dev

```bash
uv run nemo            # run the LangGraph agent (LLM demo)
uv run ruff check .    # lint
uv run mypy src        # type check
uv run pytest          # tests
```

The `.env` file configures the LLM proxy (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`).
