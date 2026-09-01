"""Minimal weather-based clothing recommendation agent."""

import json
import os
import common
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_oci import ChatOCIGenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

#-- config -------------------------------------------------------------------

load_dotenv()

# -- get_current_weather -----------------------------------------------------
@tool
def get_current_weather(city: str) -> str:
    """Get current weather in a city. Use City,CountryCode when helpful."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return "OPENWEATHER_API_KEY is not configured."
    url = "https://api.openweathermap.org/data/2.5/weather?" + urlencode(
        {"q": city, "appid": key, "units": "metric"}
    )
    try:
        with urlopen(url, timeout=10) as response:
            data = json.load(response)
        if int(data.get("cod", 200)) != 200:
            return f"Weather lookup failed: {data.get('message', 'unknown location')}"
        weather = data["weather"][0]
        main = data["main"]
        return json.dumps({
            "location": f"{data['name']}, {data.get('sys', {}).get('country', '')}".rstrip(", "),
            "temperature_c": main["temp"],
            "feels_like_c": main["feels_like"],
            "condition": weather.get("description", weather["main"]),
            "precipitation": "rain" if data.get("rain") else "snow" if data.get("snow") else "none reported",
            "wind_m_s": data.get("wind", {}).get("speed", 0),
            "humidity_percent": main["humidity"],
        })
    except HTTPError as error:
        return f"Weather lookup failed (HTTP {error.code})."
    except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        return "Weather lookup failed. Try a more specific city or try again later."

# -- main -------------------------------------------------------------------

tool_list = [get_current_weather]
model = ChatOCIGenAI(
    auth_type=os.getenv("AUTH_TYPE", "API_KEY"),
    model_id=os.environ["GENAI_MODEL"],
    service_endpoint=f"https://inference.generativeai.{os.environ['REGION']}.oci.oraclecloud.com",
    compartment_id=os.environ["COMPARTMENT_OCID"],
    model_kwargs={"temperature": 0},
).bind_tools(tool_list)


# -- call_model --------------------------------------------------------------
def call_model(state: MessagesState):
    prompt = (
        "You recommend practical adult clothing. Ask for a city when absent. "
        "For weather-dependent advice, call get_current_weather first. Base weather "
        "facts only on its result, then recommend layers, outerwear, footwear, and "
        "rain or sun accessories when appropriate."
    )
    return {"messages": [model.invoke([("system", prompt), *state["messages"]])]}


graph = StateGraph(MessagesState)
graph.add_node("assistant", call_model)
graph.add_node("tools", ToolNode(tool_list))
graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "assistant")
agent = graph.compile()

print("Weather Clothing Assistant (type 'quit' to exit)")
conversation = []
while (question := input("You: ").strip()).lower() not in {"quit", "exit"}:
    if question:
        conversation = agent.invoke({"messages": [*conversation, HumanMessage(question)]})["messages"]
        print(f"Agent: {conversation[-1].content}\n")
