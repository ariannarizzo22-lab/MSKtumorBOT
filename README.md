# Chondrosarcoma Bot

This is a Telegram bot that answers frequently asked questions (FAQ) about **chondrosarcoma**.  
It uses OpenAI embeddings and GPT models to provide answers based on documents stored in the `docs/` folder.  

## Features
- Upload documents (`.txt`, `.pdf`, `.docx`) into the `docs/` folder.
- Automatically indexes documents at startup.
- Responds to user questions via Telegram.
- Includes a disclaimer to remind users that information is **educational only** and not a substitute for medical advice.

## How it works
1. Start the bot on Telegram with `/start`.
2. Ask a question (e.g., "What is chondrosarcoma?").
3. The bot searches the indexed documents and generates an answer.

## Requirements
- Python 3.10+
- Dependencies in `requirements.txt`
- A valid OpenAI API key and Telegram bot token in `.env`

## Disclaimer
⚠️ This bot provides information for **educational purposes only**.  
It is **not medical advice** and should not replace consultation with a qualified healthcare professional.
