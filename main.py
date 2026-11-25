from pathlib import Path
from PIL import UnidentifiedImageError
from image_analysis import analyze_image, format_report


def main():
    image_path = input("Введите путь к изображению: ").strip().strip('"')
    if not Path(image_path).is_file():
        print("[-] Файл не найден.")
        return
    try:
        print(format_report(analyze_image(image_path)))
    except UnidentifiedImageError:
        print("[-] Указанный файл не распознан как изображение.")
    except Exception as error:
        print(f"[-] Ошибка при анализе файла: {error}")


if __name__ == "__main__":
    main()
