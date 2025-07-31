import asyncio
from aiogram import Bot, Dispatcher, types
import openai
import aiohttp
import os
import re
import uuid
import io
import base64

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

user_context = {}

async def speech_to_text(audio_bytes):
    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = f"{uuid.uuid4()}.ogg"
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_buffer)
        return transcript["text"]
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return None

async def text_to_speech(text):
    try:
        response = openai.Audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=text
        )
        audio_buffer = io.BytesIO()
        response.stream_to_file(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        print(f"TTS error: {e}")
        return None

async def analyze_medical_image(image_bytes):
    encoded_string = base64.b64encode(image_bytes).decode()
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "Ты врач-ассистент, подробно описываешь медицинские изображения и даешь рекомендации."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Проанализируй медицинское изображение, опиши находки и возможные патологии."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                ]}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Image analysis error: {e}")
        return None

def extract_links_and_images(text):
    url_pattern = r'(https?://\S+)'
    urls = re.findall(url_pattern, text)

    image_urls = [url for url in urls if re.search(r'\.(png|jpg|jpeg|gif|bmp|webp)$', url, re.IGNORECASE)]
    other_urls = [url for url in urls if url not in image_urls]

    return image_urls, other_urls

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id

    if message.content_type == 'voice':
        audio_file = await bot.get_file(message.voice.file_id)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{audio_file.file_path}") as resp:
                audio_bytes = await resp.read()
        user_text = await speech_to_text(audio_bytes)

    elif message.content_type == 'photo':
        photo_file = await bot.get_file(message.photo[-1].file_id)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{photo_file.file_path}") as resp:
                photo_bytes = await resp.read()
        reply_text = await analyze_medical_image(photo_bytes)

        await message.reply(reply_text)
        audio_response = await text_to_speech(reply_text)
        if audio_response:
            await bot.send_voice(user_id, types.InputFile(audio_response, filename=f"{uuid.uuid4()}.mp3"))

        return

    else:
        user_text = message.text

    user_context.setdefault(user_id, []).append({"role": "user", "content": user_text})
    context = user_context[user_id][-5:]

    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Ты виртуальный ассистент-врач, отвечай естественно."}] + context
        )
        reply_text = gpt_response.choices[0].message.content

        user_context[user_id].append({"role": "assistant", "content": reply_text})

        await message.reply(reply_text)

        image_urls, other_urls = extract_links_and_images(reply_text)

        for image_url in image_urls:
            await bot.send_photo(user_id, image_url)

        for url in other_urls:
            await bot.send_message(user_id, url)

        audio_response = await text_to_speech(reply_text)
        if audio_response:
            await bot.send_voice(user_id, types.InputFile(audio_response, filename=f"{uuid.uuid4()}.mp3"))

    except Exception as e:
        await message.reply(f"Ошибка обработки сообщения: {str(e)}")

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
