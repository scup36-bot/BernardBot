import telebot
import openai
import requests
import os
import re
import base64

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

# Функция для анализа изображения через GPT-4o Vision
def analyze_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты врач-ассистент, который максимально подробно и профессионально описывает медицинские изображения, а также предоставляет ссылки на ресурсы по конкретным заболеваниям."},
            {"role": "user", "content": [
                {"type": "text", "text": "Опиши максимально подробно это медицинское изображение. Дай ссылки на качественные и авторитетные источники по выявленному заболеванию или состоянию."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]}
        ]
    )

    return response.choices[0].message.content

# Обработка голосовых, текстовых сообщений и изображений
@bot.message_handler(content_types=['voice', 'text', 'photo'])
def handle_message(message):
    user_text = ""

    if message.content_type == 'voice':
        voice_info = bot.get_file(message.voice.file_id)
        voice_file = requests.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{voice_info.file_path}")

        with open("voice.ogg", "wb") as file:
            file.write(voice_file.content)

        user_text = speech_to_text("voice.ogg")
        os.remove("voice.ogg")

    elif message.content_type == 'text':
        user_text = message.text

    elif message.content_type == 'photo':
        photo_file = bot.get_file(message.photo[-1].file_id)
        photo_content = requests.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{photo_file.file_path}").content

        with open("photo.jpg", "wb") as file:
            file.write(photo_content)

        reply_text = analyze_image("photo.jpg")
        os.remove("photo.jpg")

        bot.send_message(message.chat.id, reply_text)

        # Извлечение ссылок и картинок
        image_urls, other_urls = extract_links_and_images(reply_text)

        for image_url in image_urls:
            bot.send_photo(message.chat.id, image_url)

        for url in other_urls:
            bot.send_message(message.chat.id, url)

        text_to_speech(reply_text)
        with open("response.mp3", "rb") as audio:
            bot.send_voice(message.chat.id, audio)

        os.remove("response.mp3")
        return

    gpt_response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты умный и дружелюбный виртуальный ассистент-врач, который отвечает максимально естественно и понятно, как человек. Давай подробные ответы, рекомендации, ссылки именно по конкретным заболеваниям, резюме и напоминания, используй ссылки и картинки, если нужно."},
            {"role": "user", "content": user_text}
        ]
    )

    reply_text = gpt_response.choices[0].message.content

    bot.send_message(message.chat.id, reply_text)

    image_urls, other_urls = extract_links_and_images(reply_text)

    for image_url in image_urls:
        bot.send_photo(message.chat.id, image_url)

    for url in other_urls:
        bot.send_message(message.chat.id, url)

    text_to_speech(reply_text)
    with open("response.mp3", "rb") as audio:
        bot.send_voice(message.chat.id, audio)

    os.remove("response.mp3")

bot.polling(non_stop=True)

