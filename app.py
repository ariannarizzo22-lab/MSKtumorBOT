import os
import json
import numpy as np
import uvicorn
from typing import List, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pathlib import Path
from dotenv import load_dotenv

# Load .env (from the same folder as app.py)
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# Read env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

BOT_NAME = os.getenv("BOT_NAME", "MSK Tumor Helper Bot")
DISCLAIMER = os.getenv(
    "DISCLAIMER",
    "⚠️ This bot provides educational information only. It is NOT medical advice and should never replace consultation with a qualified healthcare professional."
)

print(f"[CONFIG] Bot name: {BOT_NAME}")
print(f"[CONFIG] Disclaimer: {DISCLAIMER[:60]}...")


# Quick debug prints (safe)
print("DEBUG OPENAI:", OPENAI_API_KEY)
print("DEBUG TELEGRAM:", TELEGRAM_TOKEN)
print("SERVER TELEGRAM TOKEN (prefix):", TELEGRAM_TOKEN[:15], "len=", len(TELEGRAM_TOKEN))




# Document loaders
from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument

# OpenAI
from openai import OpenAI

# Vector search
import faiss

import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Language handling (Detect → Translate) ----
SUPPORTED_LANGS = {"en": "English", "it": "Italian", "de": "German", "es": "Spanish", "fr": "French"}

def detect_language(text: str) -> str:
    # Return ISO 639-1 code, default 'en' on error.
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=5,
            messages=[
                {"role": "system", "content": "Return ONLY a 2-letter ISO 639-1 language code for the user text."},
                {"role": "user", "content": text[:1000]},
            ],
        )
        code = (completion.choices[0].message.content or "").strip().lower()
        return code if len(code) == 2 else "en"
    except Exception as e:
        print("[WARN] detect_language failed:", repr(e))
        return "en"

def translate_text(answer_en: str, target_lang: str) -> str:
    # Translate English answer into target_lang; preserve lists and tone.
    if target_lang == "en" or target_lang not in SUPPORTED_LANGS:
        return answer_en
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1200,
            messages=[
                {
                    "role": "system",
                    "content": ("You are a careful medical translator. Translate from English to the target language "
                                "using neutral, plain, patient-friendly wording. Preserve bullet points, line breaks, "
                                "and the disclaimer's meaning. Do NOT add new content."),
                },
                {
                    "role": "user",
                    "content": f"Target language: {SUPPORTED_LANGS[target_lang]} ({target_lang})\n\n---\n{answer_en}",
                },
            ],
        )
        return (completion.choices[0].message.content or "").strip() or answer_en
    except Exception as e:
        print("[WARN] translate_text failed:", repr(e))
        return answer_en


app = FastAPI(title="Chondrosarcoma Telegram Bot")
@app.post("/webhook/{token}", response_class=PlainTextResponse)
async def telegram_webhook(token: str, request: Request):
    if token != TELEGRAM_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    payload = await request.json()
    msg = payload.get("message") or payload.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id:
        return "ok"

    if text.lower().startswith("/start"):
        welcome = (
            f"👋 Welcome to {BOT_NAME}!\n\n"
            "I can answer common questions about chondrosarcoma based on the information provided by your care team.\n\n"
            f"{DISCLAIMER}"
        )
        send_telegram_message(chat_id, welcome, markdown=False)
        return "ok"

    if not text:
        send_telegram_message(chat_id, "Please send a question in text form.", markdown=False)
        return "ok"

    try:
        user_lang = detect_language(text)
        answer_en = generate_answer(text)
        answer_body = translate_text(answer_en, user_lang)
    except Exception as e:
        print("[ERROR] generate_answer:", repr(e))
        answer_body = "Sorry, I couldn't generate an answer right now."

    if DISCLAIMER not in answer_body:
        answer_body = f"{answer_body}\n\n{DISCLAIMER}"

    send_telegram_message(chat_id, answer_body, markdown=False)
    return "ok"

DATA_DIR = Path(__file__).parent
DOCS_DIR = DATA_DIR / "docs"


def read_txt_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    text = []
    with open(path, "rb") as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                pass
    return "\n".join(text)


def read_docx(path: Path) -> str:
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_corpus() -> List[Dict]:
    """Return list of {source, text} from files in docs."""
    pairs = []
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for p in DOCS_DIR.iterdir():
        if p.is_file():
            try:
                if p.suffix.lower() in [".txt", ".md"]:
                    txt = read_txt_md(p)
                elif p.suffix.lower() == ".pdf":
                    txt = read_pdf(p)
                elif p.suffix.lower() == ".docx":
                    txt = read_docx(p)
                else:
                    continue
                txt = txt.replace("\r", "\n").strip()
                if txt:
                    pairs.append({"source": p.name, "text": txt})
            except Exception as e:
                print(f"[WARN] Failed to read {p}: {e}")
    return pairs


def chunk_text(text: str, max_chars: int = 400, overlap: int = 50):
    """
    Generator: yields chunks without building a big list in RAM.
    Smaller chunks + small overlap to reduce memory.
    """
    text = " ".join(text.split())
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        last_dot = chunk.rfind(". ")
        if last_dot != -1 and end != n and last_dot > int(max_chars * 0.6):
            end = start + last_dot + 2
            chunk = text[start:end]
        if chunk:
            yield chunk
        # move forward with overlap
        start = max(0, end - overlap)
        if end >= n:
            break



def build_knowledge():
    corpus = load_corpus()
    dim = 1536  # text-embedding-3-small
    index = faiss.IndexFlatIP(dim)
    entries = []

    if not corpus:
        print("[INFO] No documents found in docs/")
        return index, entries

    BATCH = 16        # se la RAM è poca, prova 8
    MAX_CHUNKS = 500  # limite di sicurezza per test; alza dopo

    total_chunks = 0
    for doc in corpus:
        source = doc["source"]
        batch_chunks = []
        for chunk in chunk_text(doc["text"]):
            batch_chunks.append(chunk)
            if len(batch_chunks) == BATCH:
                # embed & add this batch
                resp = client.embeddings.create(model="text-embedding-3-small", input=batch_chunks)
                vecs = np.array([d.embedding for d in resp.data], dtype="float32")
                vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
                index.add(vecs)
                # keep metadata in sync
                entries.extend({"source": source, "chunk": c} for c in batch_chunks)
                total_chunks += len(batch_chunks)
                print(f"[INDEX] {total_chunks} chunks")
                batch_chunks = []

                if total_chunks >= MAX_CHUNKS:
                    print("[WARN] Reached MAX_CHUNKS limit for safety.")
                    return index, entries

        # flush remainder of this document
        if batch_chunks:
            resp = client.embeddings.create(model="text-embedding-3-small", input=batch_chunks)
            vecs = np.array([d.embedding for d in resp.data], dtype="float32")
            vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
            index.add(vecs)
            entries.extend({"source": source, "chunk": c} for c in batch_chunks)
            total_chunks += len(batch_chunks)
            print(f"[INDEX] {total_chunks} chunks")

            if total_chunks >= MAX_CHUNKS:
                print("[WARN] Reached MAX_CHUNKS limit for safety.")
                return index, entries

    return index, entries


def search_similar(query: str, k: int = 5):
    entries = globals().get("ENTRIES", [])
    index = globals().get("INDEX", None)

    if not entries or index is None:
        print("[WARN] INDEX/ENTRIES not initialized; returning empty results.")
        return []

    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
        emb = resp.data[0].embedding
    except Exception as e:
        print("[ERROR] embeddings.create failed:", repr(e))
        return []

    v = np.array(emb, dtype="float32")
    v /= (np.linalg.norm(v) + 1e-8)
    D, I = index.search(v.reshape(1, -1), k)

    out = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        item = entries[idx].copy()
        item["score"] = float(score)
        out.append(item)
    return out

# Build the vector index at startup
INDEX, ENTRIES = build_knowledge()
print("[INIT] Chunks indexed:", len(ENTRIES))
print("[INIT] Docs folder:", (Path(__file__).parent / "docs").resolve())



DISCLAIMER = (
    "This information is general and **does not replace** your doctor’s advice. "
    "In case of important symptoms or urgent concerns, please contact your care team or the emergency services (112)."
)

SYSTEM_PROMPT = (
    "You are a healthcare assistant that answers **in simple English (B1 level)** questions about chondrosarcoma.\n"
    "IMPORTANT RULES:\n"
    "• Use **only** the provided Extracts. If the answer is not in the Extracts, say: "
    "'I cannot answer based on the available documents.'\n"
    "• **Do not** provide personalized medical advice, diagnosis, or dosages.\n"
    "• If you detect urgent words (e.g. 'chest pain', 'shortness of breath', 'severe bleeding', "
    "'persistent high fever', 'sudden weakness', 'uncontrollable intense pain'), respond that it may be an emergency "
    "and advise to immediately contact emergency services (112).\n"
    "• Be concise: short sentences, bullet points when useful.\n"
    "• Always end with the disclaimer.\n"
)


def generate_answer(user_question: str) -> str:
    try:
        top = search_similar(user_question, k=5)
    except Exception as e:
        print("[ERROR] search_similar failed:", repr(e))
        return "Sorry, an internal error occurred while searching the documents.\n\n_" + DISCLAIMER + "_"

    if not top:
        knowledge = "No extract available."
    else:
        seen = set()
        picks = []
        for t in top:
            key = (t["source"], t["chunk"][:80])
            if key not in seen:
                seen.add(key)
                picks.append(t)
        knowledge = "\n\n".join([f"Source: {t['source']}\nText: {t['chunk']}" for t in picks])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {user_question}\n\n=== Extracts (use only these) ===\n{knowledge}"},
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR] openai completion failed:", repr(e))
        answer = "Sorry, I had a technical issue generating the answer."

    return f"{answer}\n\n_{DISCLAIMER}_"


@app.get("/", response_class=PlainTextResponse)
def health():
    return "OK"


@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != TELEGRAM_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    update = await request.json()
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = message["chat"]["id"]
    text = message.get("text", "") or ""

    if text.strip() == "/start":
        welcome = (
            "Hello! I am an informational assistant about **chondrosarcoma**.\n"
            "Type a question and I will try to answer using ONLY the documents approved by the institution.\n\n"
            + DISCLAIMER
        )
        send_telegram_message(chat_id, welcome)
        return {"ok": True}

    if not text.strip():
        send_telegram_message(chat_id, "Please send a text message with your question.")
        return {"ok": True}

    reply = generate_answer(text.strip())
    send_telegram_message(chat_id, reply, markdown=False)
    return {"ok": True}


def send_telegram_message(chat_id: int, text: str, markdown: bool = False):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if markdown:
        payload["parse_mode"] = "Markdown"
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if not resp.ok:
            print("[WARN] Telegram send failed:", resp.status_code, resp.text)
    except Exception as e:
        print("[ERROR] Telegram send exception:", repr(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
