import os, asyncio, time, sqlite3, tempfile
from datetime import datetime, timedelta
from typing import List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Update, Message, BufferedInputFile
from aiogram.filters import Command

import httpx

# ================== НАСТРОЙКИ (ENV) ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET     = os.getenv("WEBHOOK_SECRET", "secret123")

DELIVERY_MODE = os.getenv("DELIVERY_MODE", "auto").lower()   # auto|chunks
PSEUDOSTREAMING_ENABLED = os.getenv("PSEUDOSTREAMING_ENABLED", "true").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Память по токенам (приблизительная оценка)
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "8000"))
MEMORY_TTL_DAYS    = int(os.getenv("MEMORY_TTL_DAYS", "30"))

# TTS Azure (опционально)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "azure")
TTS_REGION   = os.getenv("TTS_REGION", "")
TTS_KEY      = os.getenv("TTS_KEY", "")
TTS_VOICE    = os.getenv("TTS_VOICE", "ru-RU-DmitryNeural")
TTS_RATE     = os.getenv("TTS_RATE", "0%")
TTS_PITCH    = os.getenv("TTS_PITCH", "0%")

# ================== БОТ / FASTAPI ==================
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")

# Initialize the bot with default properties instead of using the removed
# `parse_mode` argument. In aiogram>=3.7, passing parse_mode to Bot initializer
# is not supported. Use DefaultBotProperties to configure defaults.
defaults = DefaultBotProperties(parse_mode="Markdown")
bot = Bot(token=TELEGRAM_BOT_TOKEN, defaults=defaults)
dp  = Dispatcher()
app = FastAPI()

@app.get("/health")
async def health():
    return JSONResponse({"ok": True})

# ================== ПАМЯТЬ (SQLite) ==================
DB_PATH = os.path.join(os.getcwd(), "bernard.db")

def _db():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL;")
    return c

def init_db():
    with _db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY,
            chat_id INTEGER,
            role TEXT,
            content TEXT,
            ts INTEGER
        );""")
        c.execute("""CREATE TABLE IF NOT EXISTS pinned(
            id INTEGER PRIMARY KEY,
            chat_id INTEGER,
            content TEXT,
            ts INTEGER
        );""")
init_db()

# грубая оценка "токенов": 1 токен ~ 4 символа
def _rough_tokens(s: str) -> int:
    return max(1, len(s or "") // 4)

def add_message(chat_id: int, role: str, content: str):
    with _db() as c:
        c.execute("INSERT INTO messages(chat_id, role, content, ts) VALUES(?,?,?,?)",
                  (chat_id, role, content, int(time.time())))

def build_context(chat_id: int) -> List[dict]:
    since_ts = int((datetime.utcnow() - timedelta(days=MEMORY_TTL_DAYS)).timestamp())
    with _db() as c:
        rows = c.execute("SELECT role, content FROM messages WHERE chat_id=? AND ts>=? ORDER BY id ASC",
                         (chat_id, since_ts)).fetchall()
        pins = c.execute("SELECT content FROM pinned WHERE chat_id=? ORDER BY id", (chat_id,)).fetchall()
    messages = [{"role": r, "content": t} for (r, t) in rows if t]

    # токенное окно (грубо)
    total = 0
    window = []
    for m in reversed(messages):
        cost = _rough_tokens(m["content"]) + 8
        if total + cost > CONTEXT_MAX_TOKENS:
            break
        window.append(m)
        total += cost
    window.reverse()

    prefix = [{"role": "system", "content": p[0]} for p in pins]
    return [*prefix, *window]

def get_context_preview(chat_id: int) -> str:
    with _db() as c:
        last = c.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT 5",
                         (chat_id,)).fetchall()
        pins = c.execute("SELECT id, content FROM pinned WHERE chat_id=? ORDER BY id", (chat_id,)).fetchall()
    out = ["*Pinned:*"]
    if not pins:
        out.append("—")
    else:
        for pid, content in pins:
            out.append(f"{pid}. {content}")
    out.append("\n*Последние сообщения:*")
    for r, t in last[::-1]:
        t_short = (t[:140] + "…") if len(t) > 140 else t
        out.append(f"{r}: {t_short}")
    return "\n".join(out)

def pin_fact(chat_id: int, content: str) -> int:
    with _db() as c:
        cur = c.execute("INSERT INTO pinned(chat_id, content, ts) VALUES(?,?,?)",
                        (chat_id, content, int(time.time())))
        return cur.lastrowid

def unpin_fact(chat_id: int, pid: int) -> bool:
    with _db() as c:
        cur = c.execute("DELETE FROM pinned WHERE chat_id=? AND id=?", (chat_id, pid))
        return cur.rowcount > 0

def reset_chat(chat_id: int):
    with _db() as c:
        c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))

# ================== LLM ==================
SYSTEM_PROMPT = "Ты — клинический ассистент-травматолог. Отвечай структурированно и практично."

async def llm_generate(messages: List[dict]) -> str:
    if not OPENAI_API_KEY:
        return "У меня нет OPENAI_API_KEY, поэтому отвечаю кратко: " + (messages[-1]["content"] if messages else "")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    def _call():
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT}, *messages],
            temperature=0.4,
        )
        return r.choices[0].message.content
    return await asyncio.to_thread(_call)

# ================== Доставка / Псевдостриминг ==================
MAX_MSG = 3800  # запас к лимиту 4096

def _split_chunks(text: str, limit: int = MAX_MSG) -> List[str]:
    parts, buf, cur = [], [], 0
    for line in (text or "").splitlines():
        line = line.rstrip()
        add = len(line) + 1
        if cur + add > limit:
            chunk = "\n".join(buf).strip()
            if chunk:
                if chunk.count("```") % 2 == 1:
                    chunk += "\n```"
                parts.append(chunk)
            buf, cur = [line], add
        else:
            buf.append(line); cur += add
    if buf:
        chunk = "\n".join(buf).strip()
        if chunk.count("```") % 2 == 1:
            chunk += "\n```"
        parts.append(chunk)
    return parts

async def edit_streaming(draft: Message, text: str, step: int = 800, delay: float = 0.8):
    pos = 0
    n = len(text)
    while pos < n:
        pos_next = min(n, pos + step)
        snippet = text[:pos_next]
        try:
            await draft.edit_text(snippet[:MAX_MSG], disable_web_page_preview=True)
        except Exception:
            pass
        pos = pos_next
        await asyncio.sleep(delay)

async def deliver(chat_id: int, text: str):
    mode = DELIVERY_MODE
    if mode == "chunks":
        parts = _split_chunks(text)
        for i, chunk in enumerate(parts):
            await bot.send_message(chat_id, chunk, disable_web_page_preview=True)
            if i < len(parts) - 1:
                await asyncio.sleep(1.05)
        return
    # auto
    n = len(text or "")
    if n <= 2000:
        await bot.send_message(chat_id, text, disable_web_page_preview=True)
    elif n <= 10000:
        parts = _split_chunks(text)
        for i, chunk in enumerate(parts):
            await bot.send_message(chat_id, chunk, disable_web_page_preview=True)
            if i < len(parts) - 1:
                await asyncio.sleep(1.05)
    else:
        from io import BytesIO
        buf = BytesIO((text or "").encode("utf-8"))
        buf.name = "bernard_full.md"
        await bot.send_document(chat_id, buf, caption="Полная версия")

# ================== TTS (Azure, опционально) ==================
async def azure_tts(text: str):
    if not (TTS_PROVIDER == "azure" and TTS_KEY and TTS_REGION):
        return None
    ssml = f'''<speak version="1.0" xml:lang="ru-RU">
  <voice name="{TTS_VOICE}">
    <prosody rate="{TTS_RATE}" pitch="{TTS_PITCH}">{text}</prosody>
  </voice>
</speak>'''
    url = f"https://{TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": TTS_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "ogg-48khz-16bit-mono-opus",
        "User-Agent": "bernard-bot"
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, content=ssml.encode("utf-8"), headers=headers)
        r.raise_for_status()
        data = r.content
    return data  # bytes

# ================== STT (OpenAI Whisper) ==================
async def transcribe_voice_ogg(ogg_bytes: bytes) -> str:
    """
    Распознаём голос из Telegram voice (OGG/OPUS) через OpenAI Whisper.
    Нужен OPENAI_API_KEY. Возвращает распознанный текст или пустую строку.
    """
    if not OPENAI_API_KEY:
        return ""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(ogg_bytes)
            tmp.flush()
            tmp_path = tmp.name
        def _call():
            with open(tmp_path, "rb") as f:
                r = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            return r.text
        text = await asyncio.to_thread(_call)
        return text or ""
    except Exception:
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# ================== HANDLERS ==================
router = Router()

@router.message(Command("start", "help"))
async def cmd_start(message: Message):
    text = (
        "Привет! Я Бернард.\n"
        "• Автодоставка длинных ответов (auto)\n"
        "• Псевдостриминг\n"
        "• Память по токенам + pinned\n\n"
        "Команды:\n"
        "/context — показать контекст\n"
        "/pin текст — закрепить факт\n"
        "/unpin id — открепить\n"
        "/remember текст — сохранить заметку\n"
        "/forget id — удалить заметку\n"
        "/reset — очистить историю\n"
    )
    await message.answer(text)

@router.message(Command("context"))
async def cmd_context(message: Message):
    await message.answer(get_context_preview(message.chat.id), disable_web_page_preview=True)

@router.message(Command("pin"))
async def cmd_pin(message: Message):
    parts = (message.text or "").split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        return await message.answer("Использование: /pin текст")
    pid = pin_fact(message.chat.id, parts[1].strip())
    await message.answer(f"Закреплено (id={pid}).")

@router.message(Command("unpin"))
async def cmd_unpin(message: Message):
    parts = (message.text or "").split(" ", 1)
    if len(parts) < 2 or not parts[1].strip().isdigit():
        return await message.answer("Использование: /unpin id")
    ok = unpin_fact(message.chat.id, int(parts[1]))
    await message.answer("Откреплено." if ok else "Не найдено.")

@router.message(Command("remember"))
async def cmd_remember(message: Message):
    parts = (message.text or "").split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        return await message.answer("Использование: /remember текст")
    add_message(message.chat.id, "note", parts[1].strip())
    await message.answer("Сохранено.")

@router.message(Command("forget"))
async def cmd_forget(message: Message):
    return await message.answer("Заметки удаляются вручную из DB (упростили код). Используй /unpin для закреплённых.")

@router.message(Command("reset"))
async def cmd_reset(message: Message):
    reset_chat(message.chat.id)
    await message.answer("История чата очищена.")

@router.message(F.voice)
async def on_voice(message: Message):
    # 1) скачиваем голосовой файл с Telegram
    file = await bot.get_file(message.voice.file_id)
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file.file_path}"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(file_url)
        resp.raise_for_status()
        ogg_bytes = resp.content

    # 2) STT — распознаём текст
    recognized = await transcribe_voice_ogg(ogg_bytes)
    user_text = recognized if recognized else "[voice message]"
    add_message(message.chat.id, "user", user_text)

    # 3) Готовим ответ
    ctx = build_context(message.chat.id)
    if PSEUDOSTREAMING_ENABLED:
        draft = await message.answer("Секунду, обрабатываю…")
        text = await llm_generate(ctx)
        await edit_streaming(draft, text)
    else:
        text = await llm_generate(ctx)
        await message.answer("Готовлю ответ…")

    add_message(message.chat.id, "assistant", text)

    # 4) Доставка текста
    await deliver(message.chat.id, text)

    # 5) Озвучка (если включено)
    data = await azure_tts(text)
    if data:
        await message.answer_voice(BufferedInputFile(data, "speech.ogg"), caption="Озвучка")

@router.message()
async def on_message(message: Message):
    user_text = message.text or ""
    add_message(message.chat.id, "user", user_text)
    ctx = build_context(message.chat.id)

    if PSEUDOSTREAMING_ENABLED:
        draft = await message.answer("Готовлю ответ…")
        text = await llm_generate(ctx)
        await edit_streaming(draft, text)
    else:
        text = await llm_generate(ctx)

    add_message(message.chat.id, "assistant", text)
    await deliver(message.chat.id, text)

dp.include_router(router)

# ================== WEBHOOK РОУТ ==================
@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    update = Update.model_validate(data)
    await dp.feed_update(bot, update)
    return {"ok": True}

# ================== CLI: set-webhook ==================
async def _set_webhook(public_url: str):
    url = f"{public_url}/webhook/{WEBHOOK_SECRET}"
    ok = await bot.set_webhook(url)
    print("Set webhook:", ok)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "set-webhook":
        asyncio.run(_set_webhook(sys.argv[2]))
    else:
        print("Использование:\n  python main.py set-webhook https://YOUR_PUBLIC_HTTPS_URL")