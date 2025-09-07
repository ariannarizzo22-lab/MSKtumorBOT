Chondrosarcoma Telegram Bot (RAG) — Starter

A Telegram bot that answers patient FAQs about chondrosarcoma using only the documents you place in docs/.
It performs retrieval-augmented generation (RAG) with OpenAI embeddings + FAISS.
If the answer isn’t in your docs, the bot says so (no guessing).

1) What you need

Telegram Bot Token (create via @BotFather in Telegram).

OpenAI API key (OPENAI_API_KEY).

Your documents in docs/ (PDF, DOCX, MD, TXT).

For Telegram delivery: either

Local test with a public tunnel (e.g., ngrok), or

A small HTTPS host (e.g., Render, Railway).

2) Project structure
telegram-chondrosarcoma-bot/
├─ app.py              # FastAPI server + Telegram webhook + RAG logic
├─ requirements.txt    # Python dependencies
├─ .env.example        # Copy to .env and fill keys
├─ README.md           # This file
└─ docs/               # Put your patient-friendly content here

3) Install & run locally

Requires Python 3.10+ recommended.

# From the project folder
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Prepare environment variables
cp .env.example .env
# Open .env and set:
# OPENAI_API_KEY=sk-...
# TELEGRAM_TOKEN=123456789:ABCDEF_your_telegram_bot_token

# Start the server (default: http://127.0.0.1:8000)
python app.py


Now the API is running locally. Next, expose it publicly and set the Telegram webhook.

4) Expose locally (ngrok) & set webhook
# In another terminal window
ngrok http 8000


Copy the HTTPS URL shown by ngrok (e.g., https://something.ngrok.io).

Set the Telegram webhook (replace <TOKEN> and <PUBLIC_URL>):

curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook?url=<PUBLIC_URL>/webhook/<TOKEN>"


Tip: The path /webhook/<TOKEN> acts as a simple “secret” route.

Open Telegram, find your bot, and send /start.

5) Deploy to Render (quick free option)

Push this folder to GitHub.

Create a Web Service on render.com
.

Environment Variables:

OPENAI_API_KEY: your key

TELEGRAM_TOKEN: your Telegram bot token

Start command:

uvicorn app:app --host 0.0.0.0 --port $PORT


After Render finishes deploying, set the webhook:

curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://<your-render-app>.onrender.com/webhook/<TOKEN>"

6) Add or update content

Put your files in docs/ and restart the server/service.

Supported formats: .pdf, .docx, .md, .txt.

Author good patient content:

Simple English (B1), short sentences.

Clear headings (Diagnosis, Treatment by grade, Follow-up, Prognosis, When to call 112, etc.).

Put “Last updated: YYYY-MM-DD” at the top.

7) Safety & tone (already coded)

No personalized medical advice, diagnosis, or dosages.

Emergency keywords trigger guidance to call 112 or the care team.

Every reply ends with a disclaimer:

“This information is general and does not replace your doctor’s advice. In case of important symptoms or urgent concerns, please contact your care team or the emergency services (112).”

Welcome message (/start) explains scope and limits.

You can change both in app.py:

DISCLAIMER = "..."

the /start welcome text in the Telegram handler.

8) Commands & usage

/start — Sends a welcome and disclaimer.

Normal messages — The bot searches your docs and answers in simple English using only retrieved extracts.

9) Troubleshooting

Bot doesn’t reply on Telegram

Check the webhook is set to your current URL.

Check the server logs for errors (terminal/Render logs).

Verify TELEGRAM_TOKEN is correct.

“Missing OPENAI_API_KEY/TELEGRAM_TOKEN”

Ensure .env exists with both keys set.

If deploying, add them as environment variables in the host dashboard.

No answers / “I cannot answer based on the available documents.”

Add more content to docs/.

Use clearer headings and shorter sections.

Restart the app so the index rebuilds.

PDF text not extracted

Some scanned PDFs lack embedded text. Provide a .docx or .txt version.

10) Validation checklist before sharing with patients

 Add your finalized FAQ docs in docs/.

 Try at least 20 real questions; confirm answers are correct and safe.

 Update welcome & disclaimer with your institution’s contacts (phone/email).

 Confirm red-flag phrasing matches your clinic protocol.

 Share the bot only after internal approval.

11) Switching models (optional)

Embeddings: default text-embedding-3-small (fast, cheap).
You can switch to text-embedding-3-large in app.py for higher recall.

Chat model: default gpt-4o-mini. You can swap if desired.

12) Security & privacy

Do not log or store patient-identifiable data.

Keep logs minimal and compliant with your hosting policies.

Make scope explicit: the bot provides general information only.
