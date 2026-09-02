"""A writer and reviewer agent team that produces quality-checked Wikipedia documents."""

import os
import common
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI
from pydantic import BaseModel

from tools import get_wikipedia_page

#-- config -------------------------------------------------------------------

load_dotenv()
MAX_REVISIONS = 3

class DocumentReview(BaseModel):
    approved: bool
    feedback: str

# -- write_document ----------------------------------------------------------
def write_document(request: str) -> str:
    result = common.trace_agent(
        writer_agent,
        {"messages": [HumanMessage(request)]},
        label="Writer",
    )
    return str(result["messages"][-1].content)


# -- review_document ---------------------------------------------------------
def review_document(document: str) -> DocumentReview:
    result = common.trace_agent(
        reviewer_agent,
        {"messages": [HumanMessage(document)]},
        label="Reviewer",
    )
    review = result["structured_response"]
    return review if isinstance(review, DocumentReview) else DocumentReview.model_validate(review)

# -- main --------------------------------------------------------------------
model = ChatOCIGenAI(
    auth_type=os.getenv("AUTH_TYPE", "API_KEY"),
    model_id=os.environ["GENAI_MODEL"],
    provider="generic",
    service_endpoint=f"https://inference.generativeai.{os.environ['REGION']}.oci.oraclecloud.com",
    compartment_id=os.environ["COMPARTMENT_OCID"],
    model_kwargs={"temperature": 0},
)
writer_agent = create_agent(
    model,
    [get_wikipedia_page],
    system_prompt=(
        "Write a concise, factual Markdown document from the requested English Wikipedia page. "
        "Always call get_wikipedia_page first. Use a clear title, short introduction, logical "
        "section headings, and a brief conclusion. Do not add facts not present in the page. "
        "Return only the document."
    ),
)
reviewer_agent = create_agent(
    model,
    tools=[],
    # Grok on OCI rejects LangChain's provider-native `response_format` schema.
    # Its function-calling interface does support this structured response.
    response_format=ToolStrategy(DocumentReview),
    system_prompt=(
        "Review the supplied Markdown document for grammar and structure only. Approve it only "
        "when it has correct grammar, a clear title, coherent sections, and a conclusion. Return "
        "specific revision feedback when it does not meet those requirements."
    ),
)


print("Wikipedia Writer Team (type 'quit' to exit)")
while (topic := input("Wikipedia page: ").strip()).lower() not in {"quit", "exit"}:
    if not topic:
        continue
    draft = write_document(f"Write a document about the Wikipedia page titled: {topic}")
    for attempt in range(MAX_REVISIONS):
        review = review_document(draft)
        if review.approved:
            print(f"\nApproved document:\n\n{draft}\n")
            break
        if attempt == MAX_REVISIONS - 1:
            print(f"\nDocument was not approved after {MAX_REVISIONS} reviews: {review.feedback}\n")
            break
        draft = write_document(
            "Revise this document using the review feedback. Return only the improved document.\n\n"
            f"DOCUMENT:\n{draft}\n\nREVIEW FEEDBACK:\n{review.feedback}"
        )
