import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=300,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()
    chunks = split_transcript(transcript)

    if len(chunks) <= 1:
        # Direct summarization for single-chunk transcripts
        direct_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert executive meeting summarizer. Produce a clear, professional, "
                    "high-level executive summary of this meeting transcript. Use bullet points for key takeaways, "
                    "and highlight major discussion themes.",
                ),
                ("human", "{text}"),
            ]
        )
        chain = direct_prompt | llm | StrOutputParser()
        return chain.invoke({"text": transcript})

    # Map-Reduce for longer transcripts
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely, capturing all important facts and nuances."),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert executive meeting summarizer. Synthesize these section summaries "
                "into one unified, polished executive meeting summary with comprehensive bullet points and structured sections.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()
    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript excerpt, generate an impactful, professional meeting title "
                    "(max 8 words). Return ONLY the title with no quotes or extra text.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    # Use first 3000 chars for title generation
    return title_chain.invoke(transcript[:3000]).strip('"\n ')
