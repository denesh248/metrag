# ⚡ MetRAG — Intelligent Video & Meeting Assistant

**MetRAG** is an end-to-end multimodal meeting intelligence platform. It processes YouTube videos and uploaded meeting recordings (MP4, MKV, WAV, MP3), standardizes audio via FFmpeg, generates speech-to-text using Whisper, synthesizes structured executive summaries and action items using Groq LPU™ inference, and builds an interactive Retrieval-Augmented Generation (RAG) vector database with ChromaDB for real-time question answering.

---

## ✨ Features

- **Multimodal Audio/Video Ingestion:** Ingest directly from YouTube URLs or upload local media files (`.mp4`, `.mkv`, `.mov`, `.wav`, `.mp3`, `.m4a`).
- **High-Speed Whisper STT:** Audio chunking with speech-to-text powered by Groq Cloud (`whisper-large-v3`) with local and OpenAI API fallbacks.
- **Sub-Second Groq Reasoning:** Powered by Groq LPUs (`qwen/qwen3.8-27b`) delivering instant executive briefs, delegated action items, and key decisions.
- **Conversational RAG Q&A:** ChromaDB vector index with `all-MiniLM-L6-v2` dense embeddings, grounded by strict anti-hallucination prompts.
- **Cyber-Glassmorphism UI:** Modern Streamlit interface featuring live multi-stage telemetry, task checklists, transcript keyword search, and export options.
- **Executive Export Suite:** Export meeting briefs as structured Markdown (`.md`) or structured JSON.

---

## 🛠️ Tech Stack

- **Speech-to-Text:** Whisper (`whisper-large-v3`, `whisper-1`, local)
- **LLM Reasoning:** Groq LPU™ (`qwen/qwen3.8-27b`, LLaMA models)
- **RAG & Vector Store:** LangChain LCEL, ChromaDB, HuggingFace (`all-MiniLM-L6-v2`)
- **Audio Processing:** yt-dlp, FFmpeg, PyDub
- **Frontend:** Streamlit

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/denesh248/metrag.git
cd metrag
```

### 2. Set up virtual environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r Requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the root directory (or copy `.env.example`):
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
WHISPER_MODEL=base
```

### 5. Launch the application
```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Fork or push this repository to your GitHub account (`denesh248/metrag`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select your repository: `denesh248/metrag`, branch: `main`, main file path: `app.py`.
4. Under **Advanced settings -> Secrets**, paste:
```toml
GROQ_API_KEY = "your_groq_api_key"
OPENAI_API_KEY = "your_openai_api_key"
GROQ_MODEL = "qwen/qwen3.8-27b"
```
5. Click **Deploy!**

---

## 📄 License
MIT License.
