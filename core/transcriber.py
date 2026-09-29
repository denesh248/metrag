import os
import io
import sys

# Python 3.13+ compatibility shim for audioop / pydub
try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop
        sys.modules["audioop"] = audioop
        sys.modules["pyaudioop"] = audioop
    except ImportError:
        pass

from pydub import AudioSegment

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

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
_local_whisper_model = None


def load_local_whisper():
    """Load local Whisper model (cached)."""
    global _local_whisper_model
    if _local_whisper_model is None:
        import whisper
        print(f"Loading local Whisper model: '{WHISPER_MODEL}'...")
        _local_whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Local Whisper model loaded successfully.")
    return _local_whisper_model


def transcribe_chunk_local(chunk_path: str, language: str = None) -> str:
    """Transcribe a chunk using local OpenAI Whisper."""
    model = load_local_whisper()
    kwargs = {"task": "transcribe"}
    if language and language.lower() not in ["auto", "all", ""]:
        kwargs["language"] = language
    result = model.transcribe(chunk_path, **kwargs)
    return result.get("text", "").strip()


def transcribe_chunk_groq(chunk_path: str, language: str = None) -> str:
    """
    Transcribe a chunk using Groq Whisper API (whisper-large-v3).
    Lightning fast (< 2 seconds per chunk).
    """
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)

    # Groq API accepts max 25MB. Ensure WAV is reasonable or compress to mp3 if needed
    file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
    file_to_send = chunk_path
    temp_mp3 = None

    try:
        if file_size_mb > 24:
            # Compress to MP3
            temp_mp3 = f"{chunk_path}_compressed.mp3"
            audio = AudioSegment.from_file(chunk_path)
            audio.export(temp_mp3, format="mp3", bitrate="64k")
            file_to_send = temp_mp3

        with open(file_to_send, "rb") as audio_file:
            kwargs = {
                "file": (os.path.basename(file_to_send), audio_file.read()),
                "model": "whisper-large-v3",
                "response_format": "text",
            }
            if language and language.lower() not in ["auto", "all", ""]:
                kwargs["language"] = language

            transcript = client.audio.transcriptions.create(**kwargs)
            return str(transcript).strip()
    finally:
        if temp_mp3 and os.path.exists(temp_mp3):
            os.remove(temp_mp3)


def transcribe_chunk_openai(chunk_path: str, language: str = None) -> str:
    """Transcribe a chunk using OpenAI Whisper API (whisper-1)."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    with open(chunk_path, "rb") as audio_file:
        kwargs = {
            "file": audio_file,
            "model": "whisper-1",
        }
        if language and language.lower() not in ["auto", "all", ""]:
            kwargs["language"] = language
        response = client.audio.transcriptions.create(**kwargs)
        return response.text.strip()


def transcribe_chunk(chunk_path: str, engine: str = "groq", language: str = None) -> str:
    """
    Route transcription of one chunk to the desired Whisper engine:
    - 'groq' (default): Groq cloud whisper-large-v3 (ultra-fast)
    - 'openai': OpenAI cloud whisper-1
    - 'local': Local openai-whisper
    With graceful fallback if cloud API has issues.
    """
    preferred_engine = (engine or "groq").lower()

    if preferred_engine == "groq" and os.getenv("GROQ_API_KEY"):
        try:
            return transcribe_chunk_groq(chunk_path, language=language)
        except Exception as e:
            print(f"[!] Groq Whisper failed ({e}), attempting fallback...")

    if (preferred_engine == "openai" or preferred_engine == "groq") and os.getenv("OPENAI_API_KEY"):
        try:
            return transcribe_chunk_openai(chunk_path, language=language)
        except Exception as e:
            print(f"[!] OpenAI Whisper failed ({e}), attempting local fallback...")

    # Fallback to local Whisper
    return transcribe_chunk_local(chunk_path, language=language)


def transcribe_all(chunks: list, engine: str = "groq", language: str = None) -> str:
    """Transcribe all chunks sequentially and join the resulting text."""
    full_transcript = []
    total = len(chunks)
    print(f"Transcribing {total} chunk(s) using Whisper (engine: {engine})...")

    for i, chunk in enumerate(chunks):
        print(f"  -> Processing chunk {i + 1}/{total}...")
        text = transcribe_chunk(chunk, engine=engine, language=language)
        if text:
            full_transcript.append(text)

    complete_text = " ".join(full_transcript).strip()
    print(f"Transcription complete ({len(complete_text)} characters).")
    return complete_text
