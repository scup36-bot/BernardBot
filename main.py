import telebot
import openai
import requests
import os
import re

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

# Функция для поиска ссылок и картинок в тексте
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

# Обработка голосовых и текстовых сообщений
@bot.message_handler(content_types=['voice', 'text'])
def handle_message(message):

    if message.content_type == 'voice':
        voice_info = bot.get_file(message.voice.file_id)
        voice_file = requests.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{voice_info.file_path}")

        with open("voice.ogg", "wb") as file:
            file.write(voice_file.content)

        user_text = speech_to_text("voice.ogg")
        os.remove("voice.ogg")

    elif message.content_type == 'text':
        user_text = message.text

    # Генерация ответа через GPT-4o (общий характер ответов)
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты умный и дружелюбный виртуальный ассистент. Отвечай подробно и понятно на русском языке. Если нужно добавить картинку или ссылку, пиши их отдельной строкой, используя только корректные ссылки."},
            {"role": "user", "content": user_text}
        ]
    )

    reply_text = gpt_response.choices[0].message.content

    bot.send_message(message.chat.id, reply_text)

    # Извлечение ссылок и картинок
    image_urls, other_urls = extract_links_and_images(reply_text)

    # Отправка картинок
    for image_url in image_urls:
        bot.send_photo(message.chat.id, image_url)

    # Отправка ссылок
    for url in other_urls:
        bot.send_message(message.chat.id, url)

    # Генерация голосового ответа
    text_to_speech(reply_text)
    with open("response.mp3", "rb") as audio:
        bot.send_voice(message.chat.id, audio)

    os.remove("response.mp3")

bot.polling(non_stop=True)


