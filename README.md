# Bernard — Telegram Assistant Bot (aiogram v3 + FastAPI)

**Что умеет**
- Автодоставка длинных ответов: `auto` → одно сообщение / чанки / файл
- Псевдостриминг (редактирование черновика каждые ~0.8 сек)
- Память по токенам (SQLite `bernard.db`) + pinned‑факты
- Озвучка (Azure TTS, мужской голос `ru-RU-DmitryNeural`, опционально)
- Распознавание голоса (OpenAI Whisper) из Telegram voice (OGG/OPUS)
 - Обработка изображений: принимает фото, конвертирует его в оттенки серого и отправляет краткое описание снимка с помощью GPT‑4o Vision
 - Поиск статей: команда `/articles тема` ищет в CrossRef и возвращает DOI‑ссылки

---

## Быстрый старт локально

### 1) Установить зависимости
```bash
pip install -r requirements.txt
```

### 2) Задать переменные окружения
Обязательно:
```bash
export TELEGRAM_BOT_TOKEN=ВАШ_ТОКЕН_ОТ_BOTFATHER
export WEBHOOK_SECRET=secret123
```

Опционально:
```bash
# LLM (для нормальных ответов и STT/TTS)
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini

# Озвучка (Azure TTS)
export TTS_PROVIDER=azure
export TTS_REGION=westeurope
export TTS_KEY=ВАШ_КЛЮЧ_АЗУР
export TTS_VOICE=ru-RU-DmitryNeural
export TTS_RATE=0%
export TTS_PITCH=0%

# Поведение доставки и контекста (по умолчанию и так ок)
export DELIVERY_MODE=auto
export PSEUDOSTREAMING_ENABLED=true
export CONTEXT_MAX_TOKENS=8000
export MEMORY_TTL_DAYS=30
```

### 3) Запуск приложения
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Прописать webhook (после появления публичного HTTPS‑URL)
```bash
python main.py set-webhook https://ВАШ_ПУБЛИЧНЫЙ_URL
```

---

## Деплой на Render (кратко)
1. Создайте **Web Service** (Python или Docker), порт `8000`, Healthcheck: `/health`.
2. В **Environment** добавьте переменные из шага 2.
3. Задеплойте.
4. Выполните команду из шага «webhook» с вашим публичным URL сервиса.

---

## Команды в Telegram
- `/start`, `/help` — помощь
- `/context` — показать pinned + последние реплики
- `/pin текст` — закрепить факт
- `/unpin id` — открепить факт
- `/remember текст` — заметка
- `/forget id` — удалить заметку
- `/reset` — очистить историю
 - `/articles запрос` — поиск и выдача ссылок на статьи по заданной теме

---

## Поведение без команд
Бот отвечает на **любые текстовые сообщения**.  
Голосовые: распознаёт речь (Whisper) → отвечает текстом → при включённом TTS присылает озвучку.  
Фотографии: загруженные изображения обрабатываются (например, переводятся в оттенки серого) и отправляются обратно.