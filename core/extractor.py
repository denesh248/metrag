import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert executive meeting analyst. From the meeting transcript, "
        "extract all action items, commitments, and delegated responsibilities.\n\n"
        "For each item, format as:\n"
        "- **Task:** [Clear description]\n"
        "  - **Owner:** [Person or team responsible, or 'Not specified']\n"
        "  - **Deadline:** [Due date/timeline if mentioned, or 'Not specified']\n\n"
        "If no action items are detected, return 'No action items identified.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert executive meeting analyst. From the meeting transcript, "
        "extract all key decisions, agreements, consensus points, and adopted policies.\n"
        "Format as a clean bulleted list with bolded headings. "
        "If no key decisions are detected, return 'No key decisions identified.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert executive meeting analyst. From the meeting transcript, "
        "extract all unresolved questions, open debates, blockers, and topics requiring follow-up.\n"
        "Format as a numbered list with context. "
        "If none are detected, return 'No open questions identified.'"
    )
    return chain.invoke(transcript)