"""Holiday coordinator with HR FAQ and holiday-booking ReAct subagents."""

import os
import common
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_oci import ChatOCIGenAI

from tools import (
    confirm_holiday_booking,
    get_holiday_balance,
    get_current_holiday,
    propose_holiday_booking,
    search_hr_faq,
    tools_model_init,
)

#-- config -------------------------------------------------------------------

load_dotenv()

# -- main -------------------------------------------------------------------

model = ChatOCIGenAI(
    auth_type=os.getenv("AUTH_TYPE", "API_KEY"),
    model_id=os.environ["GENAI_MODEL"],
    provider="generic",
    service_endpoint=f"https://inference.generativeai.{os.environ['REGION']}.oci.oraclecloud.com",
    compartment_id=os.environ["COMPARTMENT_OCID"],
    model_kwargs={"temperature": 0},
)
tools_model_init(model)

hr_agent = create_agent(
    model,
    [search_hr_faq],
    system_prompt="Answer HR policy questions using search_hr_faq. Do not invent policy.",
)
booking_agent = create_agent(
    model,
    [propose_holiday_booking, confirm_holiday_booking, get_current_holiday, get_holiday_balance],
    system_prompt=(
        "Help with holiday bookings. For a booking request, call propose_holiday_booking "
        "to show exact dates first. Call confirm_holiday_booking only after the user "
        "explicitly confirms the proposal. Use get_current_holiday for current-booking "
        "questions and get_holiday_balance for the remaining annual leave balance. Explain "
        "that confirmed bookings are pending manager approval."
    ),
)


# -- run_subagent ------------------------------------------------------------
def run_subagent(agent, request: str, label: str) -> str:
    result = common.trace_agent(
        agent,
        {"messages": [HumanMessage(request)]},
        label=label,
        indent=2,
    )
    return str(result["messages"][-1].content)


# -- ask_hr_agent ------------------------------------------------------------
@tool
def ask_hr_agent(question: str) -> str:
    """Ask the HR FAQ specialist about leave policy, approval, or sick leave."""
    return run_subagent(hr_agent, question, "HR subagent")


# -- ask_booking_agent -------------------------------------------------------
@tool
def ask_booking_agent(request: str) -> str:
    """Ask the holiday-booking specialist to book or show the current holiday."""
    return run_subagent(booking_agent, request, "Booking subagent")


# -- send_mail ---------------------------------------------------------------
@tool
def send_mail(recipient: str, subject: str, body: str) -> str:
    """Pretend to send an email. This dummy tool does not deliver any email."""
    return f"Dummy email sent to {recipient} with subject: {subject}"


holiday_agent = create_agent(
    model,
    [ask_hr_agent, ask_booking_agent, send_mail],
    system_prompt=(
        "You coordinate holiday requests. Use ask_hr_agent for HR policy and FAQ questions. "
        "Use ask_booking_agent for booking requests or to view the current holiday. "
        "Use send_mail only when the user asks to send an email; it is a dummy tool and "
        "does not deliver real email. "
        "Never claim a holiday is booked unless the booking specialist confirms it."
    ),
)

print("Holiday Coordinator (type 'quit' to exit)")
conversation = []
while (question := input("You: ").strip()).lower() not in {"quit", "exit"}:
    if question:
        conversation = common.trace_agent(
            holiday_agent,
            {"messages": [*conversation, HumanMessage(question)]},
            label="Holiday coordinator",
        )["messages"]
        print(f"Agent: {conversation[-1].content}\n")
