"""ReAct weather clothing agent with its tool in a separate module."""

import os
import common
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

from tools import get_current_weather

load_dotenv()

# -- main -------------------------------------------------------------------

model = ChatOCIGenAI(
    auth_type=os.getenv("AUTH_TYPE", "API_KEY"),
    model_id=os.environ["GENAI_MODEL"],
    service_endpoint=f"https://inference.generativeai.{os.environ['REGION']}.oci.oraclecloud.com",
    compartment_id=os.environ["COMPARTMENT_OCID"],
    model_kwargs={"temperature": 0},
)
agent = create_agent(
    model,
    [get_current_weather],
    system_prompt=(
        "You recommend practical adult clothing. Ask for a city when absent. "
        "For weather-dependent advice, call get_current_weather first. Base weather "
        "facts only on its result, then recommend layers, outerwear, footwear, and "
        "rain or sun accessories when appropriate."
    ),
)

print("Weather Clothing ReAct Agent (type 'quit' to exit)")
conversation = []
while (question := input("You: ").strip()).lower() not in {"quit", "exit"}:
    if question:
        conversation = agent.invoke({"messages": [*conversation, HumanMessage(question)]})["messages"]
        print(f"Agent: {conversation[-1].content}\n")
