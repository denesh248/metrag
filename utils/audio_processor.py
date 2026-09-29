import os
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

import yt_dlp
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

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio with resilient multi-client fallbacks for cloud hosting."""
    output_tmpl = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    client_strategies = [
        ["android", "ios", "mweb", "web"],
        ["android"],
        ["ios"],
        ["mweb"],
        ["web_embedded", "web"],
    ]

    last_exc = None
    for clients in client_strategies:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_tmpl,
            "extractor_args": {
                "youtube": {
                    "player_client": clients
                }
            },
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            "socket_timeout": 30,
            "retries": 3,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                raw_filename = ydl.prepare_filename(info)
                base, _ = os.path.splitext(raw_filename)
                wav_filename = f"{base}.wav"

                if os.path.exists(wav_filename):
                    return wav_filename
                elif os.path.exists(raw_filename):
                    return convert_to_wav(raw_filename)
        except Exception as e:
            last_exc = e
            continue

    if last_exc:
        err_str = str(last_exc)
        if "403" in err_str or "Forbidden" in err_str:
            raise RuntimeError(
                "YouTube blocked automated downloading from this cloud hosting server (HTTP Error 403: Forbidden). "
                "Cloud server IPs (like Streamlit Cloud / AWS) are frequently blocked by YouTube anti-bot protections. "
                "Please download the video or audio to your computer and upload it directly using the 'Upload Audio/Video' tab."
            ) from last_exc
        raise last_exc

    raise RuntimeError("Unable to download YouTube video with available audio streams.")


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready -- {len(chunks)} chunk(s) created.")
    return chunks
