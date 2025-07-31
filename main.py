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

    bot.reply_to(message, f"Вы сказали: {user_text}")

    # Генерация ответа через GPT-4 Turbo (общий характер ответов)
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Ты умный и дружелюбный виртуальный ассистент. Отвечай подробно и понятно на русском языке."},
            {"role": "user", "content": user_text}
        ]
    )

    reply_text = gpt_response.choices[0].message.content
    bot.send_message(message.chat.id, reply_text)

    # Генерация голосового ответа
    text_to_speech(reply_text)
    with open("response.mp3", "rb") as audio:
        bot.send_voice(message.chat.id, audio)

    os.remove("response.mp3")
