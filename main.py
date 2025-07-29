import telebot
import openai
import requests
import os
import re
from PIL import Image
from io import BytesIO

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Функция для распознавания голоса через Whisper API
def speech_to_text(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

# Функция для генерации голоса из текста (TTS)
def text_to_speech(text, file_path="response.mp3"):
    response = openai.Audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    response.stream_to_file(file_path)

# Функция для анализа изображения через GPT-4 Vision
def analyze_image(image_url):
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Подробно опиши это изображение. Если это медицинский снимок (рентген, УЗИ и т.д.), дай профессиональный анализ."},
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content

# Функция для извлечения ссылок и изображений из текста
def extract_links_and_images(text):
    url_pattern = r'(https?://\S+)'
    urls = re.findall(url_pattern, text)

    image_urls = []
    other_urls = []

    for url in urls:
        cleaned_url = url.strip('()[]<>.,;:\"\'')
        if any(ext in cleaned_url.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']):
            image_urls.append(cleaned_url)
        else:
            other_urls.append(cleaned_url)

    return image_urls, other_urls

# Обработка голосовых, текстовых сообщений и изображений
@bot.message_handler(content_types=['voice', 'text', 'photo'])
def handle_message(message):
    user_text = ""
    image_url = None

    if message.content_type == 'voice':
        try:
            voice_info = bot.get_file(message.voice.file_id)
            voice_file = requests.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{voice_info.file_path}")

            with open("voice.ogg", "wb") as file:
                file.write(voice_file.content)

            user_text = speech_to_text("voice.ogg")
            os.remove("voice.ogg")
            
            # Добавляем пометку, что это голосовое сообщение
            user_text = f"[Голосовое сообщение]: {user_text}"
        except Exception as e:
            bot.reply_to(message, f"Ошибка обработки голосового сообщения: {str(e)}")
            return

    elif message.content_type == 'text':
        user_text = message.text

    elif message.content_type == 'photo':
        try:
            photo_file = bot.get_file(message.photo[-1].file_id)
            image_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{photo_file.file_path}"
            
            if not user_text:  # Если пользователь просто отправил фото без текста
                user_text = "Что на этом изображении?"
            
            # Анализируем изображение
            image_analysis = analyze_image(image_url)
            user_text = f"{user_text}\n\n[Изображение]: {image_url}\n[Анализ изображения]: {image_analysis}"
        except Exception as e:
            bot.reply_to(message, f"Ошибка обработки изображения: {str(e)}")
            return

    try:
        # Генерация ответа через GPT-4o (естественные, живые ответы)
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты умный и дружелюбный виртуальный ассистент-врач. Отвечай максимально естественно и понятно, как человек. Давай подробные ответы, рекомендации, резюме и напоминания. Используй ссылки и картинки, если нужно. Если тебе прислали медицинский снимок, дай профессиональный анализ."},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7
        )

        reply_text = gpt_response.choices[0].message.content

        # Отправляем текстовый ответ
        bot.send_message(message.chat.id, reply_text)

        # Извлечение ссылок и картинок из ответа
        image_urls, other_urls = extract_links_and_images(reply_text)

        # Отправка картинок из ответа
        for img_url in image_urls:
            try:
                bot.send_photo(message.chat.id, img_url)
            except:
                bot.send_message(message.chat.id, f"Не удалось отправить изображение: {img_url}")

        # Отправка ссылок из ответа
        for url in other_urls:
            bot.send_message(message.chat.id, url)

        # Генерация и отправка голосового ответа (если текст не слишком длинный)
        if len(reply_text) <= 4096:  # Ограничение TTS
            try:
                text_to_speech(reply_text)
                with open("response.mp3", "rb") as audio:
                    bot.send_voice(message.chat.id, audio)
                os.remove("response.mp3")
            except Exception as e:
                print(f"Ошибка генерации голосового ответа: {str(e)}")
        else:
            # Отправляем только часть текста как аудио, если текст слишком длинный
            try:
                text_to_speech(reply_text[:4000] + "... (текст продолжается)")
                with open("response.mp3", "rb") as audio:
                    bot.send_voice(message.chat.id, audio)
                os.remove("response.mp3")
            except Exception as e:
                print(f"Ошибка генерации голосового ответа: {str(e)}")

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка при обработке запроса: {str(e)}")

if __name__ == '__main__':
    print("Бот запущен...")
    bot.polling(non_stop=True)

