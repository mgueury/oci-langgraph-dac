# LangGraph OCI Agent Examples

Small interactive examples using OCI Generative AI, LangGraph/LangChain, and
shared tools. Each script loads credentials from `.env` and exits when you type
`quit` or `exit`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your OCI configuration, OpenWeather key, and Langfuse settings:

```env
GENAI_MODEL=xai.grok-4.20-0309-reasoning
REGION=us-chicago-1
COMPARTMENT_OCID=ocid1.compartment.oc1..your-compartment
AUTH_TYPE=API_KEY
TIMEZONE=Europe/Brussels
OPENWEATHER_API_KEY=your-openweather-key
LANGFUSE_PUBLIC_KEY=pk-lf-change
LANGFUSE_SECRET_KEY=sk-lf-change
LANGFUSE_BASE_URL=http://12.34.56.78
```

For `AUTH_TYPE=API_KEY`, configure the OCI SDK as usual in `~/.oci/config`.
`TIMEZONE` must be an IANA timezone and is used by the holiday examples for
relative dates such as “tomorrow.”

Examples 3–5 also send their agent, model, and tool activity to Langfuse when
all three `LANGFUSE_*` variables are set. Replace the placeholder keys with the
keys for your Langfuse project.

## Example 1: Basic weather graph

`ex1_weather_basic.py` shows the explicit LangGraph flow: model → tool → model.

```bash
python3 ex1_weather_basic.py
```

Try these sentences:

```text
What should I wear in Brussels today?
Should I take an umbrella in London, GB?
What clothing do I need in Tokyo, JP right now?
```

## Example 2: ReAct weather agent

`ex2_react_agent.py` provides the same weather advice with the prebuilt ReAct
agent loop instead of manually defining graph nodes and edges.

```bash
python3 ex2_react_agent.py
```

Try these sentences:

```text
What should I wear in Paris, FR?
Is it cold enough for a coat in New York, US?
What footwear should I use in Singapore?
```

## Example 3: ReAct agent with shared tools

`ex3_react_agent.py` is the ReAct version that imports `get_current_weather`
from `tools.py`, demonstrating reusable tool modules.

```bash
python3 ex3_react_agent.py
```

Try these sentences:

```text
What should I wear in Brussels, BE today?
Do I need sunglasses in Madrid, ES?
How should I dress for the weather in Sydney, AU?
```

## Example 4: Wikipedia writer and reviewer

`ex4_reflection.py` runs a writer agent and a reviewer agent. The writer uses an
English Wikipedia page as its source; the reviewer checks grammar and structure.
Only an approved Markdown document is displayed. The writer can revise it up to
three times.

```bash
python3 ex4_reflection.py
```

Enter a Wikipedia page title at the prompt:

```text
Ada Lovelace
Artificial intelligence
Brussels
```

## Example 5: Holiday supervisor

`ex5_supervisor.py` coordinates two specialist agents: one answers HR FAQ
questions and the other proposes and confirms holiday bookings. Confirmed
bookings are stored in `holiday.json`; the balance starts at 25 weekdays each
year.

```bash
python3 ex5_supervisor.py
```

Try these sentences in order:

```text
How many annual leave days do I have?
What is the approval policy for a holiday?
Book a holiday next Friday.
Confirm.
What is my holiday balance?
What are my current holidays?
```

The booking agent first shows the resolved dates. It writes to `holiday.json`
only after you explicitly confirm the proposal.

## Shared tools

`tools.py` holds reusable tools for weather, Wikipedia content, HR FAQ matching,
holiday date normalization, booking storage, and holiday-balance calculation.
`tools_model_init(model)` initializes its LLM-backed tools from the OCI model.
