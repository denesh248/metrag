import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, engine: str = "groq") -> dict:
    print("\n" + "=" * 60)
    print("[*] Starting AI Video & Meeting Assistant Pipeline")
    print("=" * 60)

    print("\n[1/6] Processing Audio...")
    chunks = process_input(source)

    print(f"\n[2/6] Transcribing with Whisper ({engine})...")
    transcript = transcribe_all(chunks, engine=engine)
    print(f"Transcript preview (first 250 chars):\n{transcript[:250]}...\n")

    print("[3/6] Generating Executive Title...")
    title = generate_title(transcript)

    print("[4/6] Summarizing Transcript with Groq...")
    summary = summarize(transcript)

    print("[5/6] Extracting Action Items, Decisions, & Open Questions...")
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    print("[6/6] Initializing Chroma Vector Store & RAG Engine...")
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("AI Video Assistant (Groq LPU + Whisper STT + RAG)")
    print("=" * 60)

    source = input("Enter YouTube URL or local media path: ").strip()
    if not source:
        print("[!] No input source provided. Exiting.")
        exit(1)

    result = run_pipeline(source)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nExecutive Summary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Interactive RAG Session
    print("\nChat with your meeting transcript (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Session ended.")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\nAssistant:\n{answer}")