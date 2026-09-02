# LangGraph OCI Agent Examples

Five small, interactive Python examples that combine OCI Generative AI with
LangChain and LangGraph. They progress from an explicit tool graph to traced
ReAct agents, a writer/reviewer team, and a holiday supervisor with subagents.

Every program reads its local configuration from `.env`. Type `quit` or `exit`
at a prompt to stop it.

## Setup

Create an isolated environment, install the dependencies, and create your local
configuration file:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your OCI configuration and OpenWeather key:

```env
GENAI_MODEL=xai.grok-4.20-0309-reasoning
REGION=us-chicago-1
COMPARTMENT_OCID=ocid1.compartment.oc1..replace-me
AUTH_TYPE=API_KEY
TIMEZONE=Europe/Brussels
OPENWEATHER_API_KEY=replace-me
```

With `AUTH_TYPE=API_KEY`, configure the OCI SDK in `~/.oci/config`. Other
authentication types supported by `langchain-oci` can be used by changing
`AUTH_TYPE`.

`TIMEZONE` must be an IANA timezone. The holiday example uses it to interpret
relative dates such as “tomorrow” and “next Friday.”

`.env` and the holiday data file are ignored by Git; do not commit real keys.

## Observability and terminal traces

Examples 3–5 print their execution flow as they run:

- Agent and subagent labels are cyan.
- Tool calls are yellow and include their arguments.
- Successful tool results are green; error-like results are red.

Set `NO_COLOR=1` to use plain terminal output.

The same calls can be recorded in Langfuse. Add all three settings to `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_BASE_URL=http://your-langfuse-host:3000
```

Tracing is enabled only when all three settings are present. The shared trace
helper attaches the Langfuse callback to each agent stream, so the dashboard
shows model and tool activity in addition to the terminal trace.

## Examples

### 1. Explicit weather graph

`ex1_weather_basic.py` builds the complete LangGraph flow explicitly:
model → weather tool → model. The weather tool is defined in the same file.

```bash
python3 ex1_weather_basic.py
```

Try: `What should I wear in Brussels today?`

### 2. ReAct weather agent

`ex2_react_agent.py` uses LangChain’s prebuilt agent loop with the reusable
weather tool from `tools.py`.

```bash
python3 ex2_react_agent.py
```

Try: `Should I take an umbrella in London, GB?`

### 3. Traced ReAct weather agent

`ex3_react_agent.py` is the same weather assistant with readable terminal and
optional Langfuse tracing.

```bash
python3 ex3_react_agent.py
```

Try: `What should I wear in Brussels, BE today?`

### 4. Wikipedia writer and reviewer

`ex4_reflection.py` runs two agents. The writer retrieves the requested English
Wikipedia page and creates a concise Markdown document. The reviewer checks its
grammar and structure, and can request up to three revisions. Writer activity,
Wikipedia calls, and reviewer phases are traced.

```bash
python3 ex4_reflection.py
```

Enter a page title, for example:

```text
Ada Lovelace
Artificial intelligence
Brussels
```

The writer only uses facts returned by the Wikipedia tool. If Wikipedia is
unreachable, the trace displays the tool error.

### 5. Holiday supervisor

`ex5_supervisor.py` coordinates an HR FAQ specialist and a holiday-booking
specialist. The supervisor delegates requests to the appropriate subagent and
traces that delegation along with the nested tool calls.

```bash
python3 ex5_supervisor.py
```

Try this sequence:

```text
How many annual leave days do I have?
What is the approval policy for a holiday?
Book a holiday next Friday.
Confirm.
What is my holiday balance?
What are my current holidays?
```

Booking is deliberately two-step: the specialist first proposes exact dates,
then writes a confirmed booking to `holiday.json` only after `Confirm`. Confirmed
bookings remain pending manager approval. The annual allowance is 25 weekdays.

The supervisor also has `send_mail`, a dummy tool that returns a confirmation
but never delivers an email.

## Shared modules

`tools.py` provides the reusable weather, Wikipedia, HR, and holiday tools.
`tools_model_init(model)` supplies the OCI model to the HR and holiday-date
tools that require it.

`common.py` contains terminal formatting, the Langfuse callback integration,
and the compatibility handling for nullable OCI token-usage details.

## Tests

Run the test suite from the activated virtual environment:

```bash
python -m unittest discover -s tests
```
