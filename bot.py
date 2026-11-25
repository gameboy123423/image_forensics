import os
import tempfile
from pathlib import Path

import telebot
from image_analysis import analyze_image, format_report


bot = telebot.TeleBot('')


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(
        message,
        "Здравствуйте! Отправьте изображение как фото или документ. "
        "Я проверю EXIF, ELA, локальные различия шума и возможные копии фрагментов."
    )


@bot.message_handler(content_types=["photo", "document"])
def handle_incoming_file(message):
    temp_path = None
    output_dir = None
    try:
        if message.content_type == "photo":
            file_info = bot.get_file(message.photo[-1].file_id)
            suffix = ".jpg"
        else:
            document = message.document
            if not document.mime_type or not document.mime_type.startswith("image/"):
                bot.reply_to(message, "Пожалуйста, отправьте файл изображения.")
                return
            file_info = bot.get_file(document.file_id)
            suffix = Path(document.file_name or "image").suffix or ".img"

        downloaded_file = bot.download_file(file_info.file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(downloaded_file)
            temp_path = temp_file.name

        output_dir = tempfile.mkdtemp(prefix="image_analysis_")
        report = analyze_image(temp_path, output_dir=output_dir)
        bot.reply_to(message, format_report(report))

        for map_path, caption in (
            (report["noise"]["map_path"], "Карта локальных различий шума"),
            (report["clone"]["map_path"], "Карта совпадений Clone Detection"),
        ):
            with open(map_path, "rb") as map_file:
                bot.send_photo(message.chat.id, map_file, caption=caption)

    except Exception as error:
        bot.reply_to(message, f"Произошла ошибка при анализе: {error}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if output_dir and os.path.isdir(output_dir):
            for file_name in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(output_dir)


if __name__ == "__main__":
    print("Бот анализа изображений запущен...")
    bot.infinity_polling()
