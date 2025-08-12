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
import urllib.parse
from PIL import Image

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

# ================== НАСТРОЙКИ РЕЖИМОВ И СТИЛЕЙ ==================
# По умолчанию Бернард работает как травматолог‑ортопед. Пользователь может
# переключать режимы через команду /mode, а также настраивать краткость
# ответов (/brief, /detailed) и желаемую креативность (/temp low|mid|high).

# Текущий режим. Возможные значения:
#  - "ortho": клинический ассистент‑травматолог (по умолчанию)
#  - "general": универсальный ассистент для любых тем
#  - "coach": инструктор по восстановлению, спорту, здоровью
#  - "translate": переводчик (двусторонний)
#  - "writer": помощник по письменным материалам и резюме
#  - "research": поиск и краткое резюме научных статей (использует существующую /articles)
MODE: str = os.getenv("DEFAULT_MODE", "ortho").lower()

# Текущая степень детализации ответа:
#  - "normal": стандартный уровень
#  - "brief": кратко (1–2 предложения)
#  - "detailed": развёрнуто и структурировано
VERBOSITY: str = "normal"

# Предпочтительная креативность (температура) ответа:
#  - "low", "mid", "high". Набор сохраняется для совместимости, но не
#    передаётся напрямую в OpenAI API, поскольку GPT‑5 и некоторые
#    версии GPT‑4o не поддерживают настройку temperature.
TEMP_PREF: str = "mid"

# Шаблоны системных сообщений для различных режимов. Эти строки будут
# дополняться указаниями на стиль (краткость/детальность) в функции
# build_system_prompt().
MODE_PROMPTS = {
    "ortho": (
        "Ты — клинический ассистент‑травматолог и ортопед. "
        "Отвечай практично: структурируй боль/жалобу, предложи тесты, возможные дифференциальные "
        "диагнозы, варианты лечения и рекомендации по обращению к врачу. Помни про дисклеймер: "
        "не ставь диагнозы и советуй обратиться к специалисту при наличии красных флагов."
    ),
    "general": (
        "Ты — универсальный цифровой ассистент. Помогаешь в любых вопросах: "
        "новости, технологии, советы по здоровью и быту, рецепты, путешествия, "
        "личная эффективность. Отвечай дружелюбно и понятно."
    ),
    "coach": (
        "Ты — инструктор по восстановлению и здоровому образу жизни. "
        "Помогаешь с упражнениями, растяжкой, планированием тренировок, "
        "профилактикой травм и рекомендациями по питанию и сну. Не даёшь медицинских диагнозов."
    ),
    "translate": (
        "Ты — переводчик. Переводи текст между русским и английским языками, "
        "сохраняй смысл, стиль и терминологию. Если требуется, можешь добавить "
        "пояснения к сложным терминам."
    ),
    "writer": (
        "Ты — помощник по письменным материалам: резюме, письма, статьи, посты. "
        "Помогаешь структурировать мысли, сохранять правильный тон (деловой, дружелюбный, "
        "нейтральный) и избегать ошибок."
    ),
    "research": (
        "Ты — исследователь. Помогаешь искать научные публикации и статьи по ключевым "
        "словам, формируешь краткие выводы и указываешь DOI/PMID, если доступны. "
        "Используй команду /articles для поиска свежих статей."
    ),
}

def build_system_prompt() -> str:
    """Собирает системный промпт в зависимости от выбранного режима и стиля."""
    base = MODE_PROMPTS.get(MODE, MODE_PROMPTS.get("general"))
    extras = []
    # Добавляем указания по краткости
    if VERBOSITY == "brief":
        extras.append("Отвечай максимально кратко (1–2 предложения).")
    elif VERBOSITY == "detailed":
        extras.append("Отвечай развёрнуто, структурируй ответ по пунктам.")
    return (base + " " + " ".join(extras)).strip()

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
        # Используем динамический системный промпт в зависимости от режима и стиля.
        # Обратите внимание: не передаём параметр temperature — GPT‑5 и некоторые модели GPT‑4o не
        # поддерживают его настройку и вернут ошибку unsupported_value, если он указан.
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": build_system_prompt()}, *messages],
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

# ================== Image Analysis (GPT-4o Vision) ==================
# Анализируем изображение при помощи модели GPT-4o. На вход подаются байты изображения,
# на выходе — текстовое описание содержимого снимка. В случае ошибки возвращается пустая строка.
async def analyze_image(image_bytes: bytes) -> str:
    """
    Use OpenAI's GPT-4o model to generate a concise description of the given image.

    The function encodes the image as base64 data URI and sends it to the model along with
    an instruction. It returns the generated description or an empty string if any error occurs.
    """
    if not OPENAI_API_KEY:
        return ""
    try:
        import base64
        from openai import OpenAI
        # Encode image to base64 data URI
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Compose messages for vision model. System prompt puts the assistant into clinical context.
        # Compose a more detailed prompt for GPT‑Vision.  The system prompt now
        # asks the assistant to act as an experienced orthopaedic surgeon and
        # describe the image in greater detail, including relevant anatomical
        # structures and any notable pathologies.  The assistant must still
        # clarify that the description is not a definitive diagnosis.
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — опытный ортопед и травматолог. Подробно и точно опиши, что изображено на медицинском снимке: "
                    "назови ключевые анатомические структуры, возможные патологические изменения и особенности. "
                    "Добавь, что это описание не является окончательным диагнозом."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Опиши детально, что изображено на снимке."},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ]
        # Use gpt-4o for vision; fallback to any model supporting images
        model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
        def _call():
            """
            Make a request to the vision model.  Some image-capable models (for example
            GPT‑4o and GPT‑5 vision) only accept the default temperature of 1.  Passing
            a custom temperature causes a 400 error (`unsupported_value`) as seen in
            the Render logs.  Therefore we omit the `temperature` parameter entirely
            and rely on the model's default.
            """
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512
            )
            return response.choices[0].message.content

        description = await asyncio.to_thread(_call)
        return description.strip() if description else ""
    except Exception:
        return ""

# ================== HANDLERS ==================
router = Router()

@router.message(Command("start", "help"))
async def cmd_start(message: Message):
    text = (
        "Привет! Я Бернард.\n"
        "• Автодоставка длинных ответов (auto)\n"
        "• Псевдостриминг\n"
        "• Память по токенам + pinned\n"
        "• Возможность получения и обработки голосовых и фото сообщений\n"
        "• Поиск статей с помощью /articles\n\n"
        "Команды:\n"
        "/context — показать контекст\n"
        "/pin текст — закрепить факт\n"
        "/unpin id — открепить\n"
        "/remember текст — сохранить заметку\n"
        "/forget id — удалить заметку\n"
        "/reset — очистить историю\n"
        "/articles запрос — найти статьи по теме\n"
        "/mode режим — переключить профиль (ortho, general, coach, translate, writer, research)\n"
        "/brief — отвечать кратко\n"
        "/detailed — отвечать развёрнуто\n"
        "/temp low|mid|high — установить уровень креативности (не влияет на GPT‑5)\n"
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

# ====== Режимы и стиль ======

@router.message(Command("mode"))
async def cmd_mode(message: Message):
    """
    Переключает режим работы бота. Допустимые режимы: ortho, general,
    coach, translate, writer, research. Если пользователь не указал аргумент,
    бот перечисляет доступные режимы.
    """
    global MODE
    parts = (message.text or "").split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        # Сообщаем о доступных режимах
        modes = ", ".join(MODE_PROMPTS.keys())
        return await message.answer(f"Доступные режимы: {modes}. Используйте /mode <имя> для переключения.")
    new_mode = parts[1].strip().lower()
    if new_mode not in MODE_PROMPTS:
        modes = ", ".join(MODE_PROMPTS.keys())
        return await message.answer(f"Неизвестный режим: {new_mode}. Допустимые: {modes}.")
    MODE = new_mode
    await message.answer(f"Режим изменён на {new_mode}.")

@router.message(Command("brief"))
async def cmd_brief(message: Message):
    """Устанавливает краткий стиль ответа."""
    global VERBOSITY
    VERBOSITY = "brief"
    await message.answer("Теперь отвечаю кратко.")

@router.message(Command("detailed"))
async def cmd_detailed(message: Message):
    """Устанавливает развёрнутый стиль ответа."""
    global VERBOSITY
    VERBOSITY = "detailed"
    await message.answer("Теперь отвечаю развёрнуто.")

@router.message(Command("temp"))
async def cmd_temp(message: Message):
    """
    Устанавливает желаемый уровень креативности (temperature). Допустимые: low, mid, high.
    Эти значения используются как подсказки и не передаются напрямую в OpenAI API
    из‑за ограничений некоторых моделей.
    """
    global TEMP_PREF
    parts = (message.text or "").split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        return await message.answer("Использование: /temp low|mid|high")
    level = parts[1].strip().lower()
    if level not in {"low", "mid", "high"}:
        return await message.answer("Недопустимое значение. Варианты: low, mid, high.")
    TEMP_PREF = level
    await message.answer(f"Предпочтительная креативность установлена на {level}.")

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
    # Если включён псевдостриминг, мы уже отредактировали сообщение `draft` и передали полный текст
    # через edit_streaming. В этом случае не отправляем текст повторно через deliver, чтобы
    # избежать дублирования ответов. Если псевдостриминг отключен, используем deliver.
    if not PSEUDOSTREAMING_ENABLED:
        await deliver(message.chat.id, text)

    # 5) Озвучка (если включено)
    data = await azure_tts(text)
    if data:
        await message.answer_voice(BufferedInputFile(data, "speech.ogg"), caption="Озвучка")

# Удалён отдельный обработчик фотографий. Теперь всё управление изображениями находится
# в основном обработчике on_message. Это предотвращает дублирование ответов и гарантирует,
# что бот не отправит текстовый ответ от LLM для сообщений с изображением.

@router.message(Command("articles"))
async def cmd_articles(message: Message):
    """
    Search for recent scholarly articles via the CrossRef API and return a few DOIs.
    Usage: /articles тема поиска
    """
    query = message.get_args().strip()
    if not query:
        return await message.answer("Укажите тему поиска, например: /articles артрит")
    # Build search URL for CrossRef: search by title; limit results
    params = {
        "query.title": query,
        "rows": "5",
        "sort": "published",
        "order": "desc",
    }
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            res = await client.get(url)
            res.raise_for_status()
            data = res.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            return await message.answer("По вашему запросу статьи не найдены.")
        lines = []
        for item in items:
            title_list = item.get("title", [])
            title = title_list[0] if title_list else "Без названия"
            doi = item.get("DOI")
            link = f"https://doi.org/{doi}" if doi else ""
            # Truncate title if too long
            if len(title) > 120:
                title = title[:117] + "…"
            lines.append(f"• {title}\n{link}")
        reply = "Вот несколько релевантных статей:\n\n" + "\n\n".join(lines)
        await message.answer(reply, disable_web_page_preview=True)
    except Exception:
        await message.answer("Не удалось получить статьи. Попробуйте снова позже.")

@router.message()
async def on_message(message: Message):
    # Если входящее сообщение содержит фотографию или документ с изображением,
    # обрабатываем его сразу и не отправляем в LLM. Это позволяет корректно
    # реагировать на пересланные фотографии, документы формата image/* и
    # сообщения с подписью, в которых присутствует изображение.
    try:
        # Проверяем, есть ли список фотографий (обычные фото)
        has_photo = getattr(message, "photo", None)
        # Проверяем, является ли документ картинкой (например, пересланный
        # снимок может приходить как документ с MIME-типа image/*)
        has_image_doc = (
            hasattr(message, "document")
            and message.document is not None
            and getattr(message.document, "mime_type", "").startswith("image/")
        )
        if has_photo or has_image_doc:
            # Выбираем объект File для загрузки: либо последняя фотография в списке,
            # либо сам документ. Далее запрашиваем файл у API Telegram и
            # конвертируем его в оттенки серого.
            if has_photo:
                photo = message.photo[-1]
                file = await bot.get_file(photo.file_id)
            else:
                file = await bot.get_file(message.document.file_id)
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file.file_path}"
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(file_url)
                resp.raise_for_status()
                image_bytes = resp.content
            import io
            img = Image.open(io.BytesIO(image_bytes))
            img_gray = img.convert("L")
            buf = io.BytesIO()
            img_gray.save(buf, format="PNG")
            buf.seek(0)
            # Сначала отправляем сообщение об успешном получении изображения и саму обработанную картинку.
            await message.answer(
                "Изображение получено. Вот версия в оттенках серого. Сейчас попробую описать его содержимое…"
            )
            await bot.send_photo(
                message.chat.id,
                BufferedInputFile(buf.getvalue(), "processed.png"),
            )
            # Далее выполняем анализ изображения с помощью GPT‑4o и отправляем результат.
            analysis = await analyze_image(image_bytes)
            if analysis:
                await message.answer(analysis)
            else:
                await message.answer(
                    "Не удалось выполнить автоматический анализ изображения. Если вас интересует конкретная структура или вопрос — уточните его, и я помогу по тексту."
                )
            # Мы не отправляем сообщение в LLM, поскольку он не понимает изображения.
            return
    except Exception:
        # В случае любой ошибки сообщаем пользователю и продолжаем работу (не вызываем LLM).
        await message.answer(
            "Не удалось обработать изображение. Попробуйте отправить другое изображение или попросите помощи."
        )
        return

    # Otherwise process the message normally via the LLM
    user_text = message.text or ""
    add_message(message.chat.id, "user", user_text)
    ctx = build_context(message.chat.id)

    if PSEUDOSTREAMING_ENABLED:
        # Создаём черновик и имитируем потоковый вывод, редактируя это сообщение.  Не
        # вызываем deliver после завершения стрима, чтобы избежать дублирования ответа.
        draft = await message.answer("Готовлю ответ…")
        text = await llm_generate(ctx)
        await edit_streaming(draft, text)
        # Добавляем запись о сообщении в память после завершения стрима
        add_message(message.chat.id, "assistant", text)
        return
    else:
        # Псевдостриминг выключен: просто генерируем и доставляем ответ
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