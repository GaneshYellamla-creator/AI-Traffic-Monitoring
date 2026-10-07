from openai import OpenAI
from dotenv import load_dotenv
from rag.rag_engine import TrafficRAG

load_dotenv()

client = OpenAI()

rag = TrafficRAG()


def analyze_event(event):

    query = f"""
    Traffic monitoring event:

    Object: {event.get("object")}
    Tracking ID: {event.get("id")}
    Direction: {event.get("direction")}
    Zone: {event.get("zone")}
    Speed: {event.get("speed")}

    What traffic safety rules are relevant to this event?
    """

    context = rag.get_context(query)

    prompt = f"""
You are an AI traffic monitoring assistant.

Use ONLY the provided knowledge-base context when
explaining traffic rules.

Knowledge Base:
{context}

Detected Event:
{event}

Provide:

1. Event summary
2. Possible safety concern
3. Relevant rule
4. Recommended action

Do not invent laws or penalties.
If the knowledge base does not contain enough
information, clearly say so.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text
