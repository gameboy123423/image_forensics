'''
Написано как курсовой проект если заинтересованы в работе и технологиях попытайтесь связаться со мной
'''

import io
import os
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageStat, ExifTags, ImageFilter
#Все что сверху это импорты



#Тут у нас кортеж с названиями эдиторов(программ для обработки изображений)
EDITOR_KEYWORDS = (
    "photoshop", "lightroom", "gimp", "paint.net", "affinity", "snapseed",
    "picsart", "canva", "pixlr", "illustrator", "capture one", "corel",
    "meitu", "facetune"
)


def _safe_text(value):
    if value is None:
        return "Не указано"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


#Тут Мы читаем метаданные используется библиотека PIL
def _read_metadata(image):
    exif = image.getexif()#Тут мы вытаскиваем метаданные в отдельную перменную
    metadata = {}
    for tag_id, value in exif.items():#Здесь мы оттдельно ищем данные по геометкам так как в ексифе они сидят не как числа
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if tag_name == "GPSInfo":
            try:
                gps_raw = exif.get_ifd(ExifTags.IFD.GPSInfo)
                metadata["GPSInfo"] = {
                    ExifTags.GPSTAGS.get(k, str(k)): v for k, v in gps_raw.items()
                }
            except Exception:
                metadata["GPSInfo"] = {}
        else:
            metadata[tag_name] = value
    return metadata


#Здесь те метаданные по геометкам которые мы вытащили превращаем в числовые значения
def _rational_to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return float(value.numerator) / float(value.denominator)

#Здесь мы их переводим в градусные значения чтобы мы могли потом в гугл мап кидать их
def _gps_to_degrees(parts, reference):
    degrees, minutes, seconds = parts
    result = (
        _rational_to_float(degrees)
        + _rational_to_float(minutes) / 60
        + _rational_to_float(seconds) / 3600
    )
    return -result if reference in ("S", "W") else result

#Здесь начинается ELA Анализ, в подробности вдаваться не буду, сами учите, мне лень вас учить
def _ela_analysis(image):
    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=90)#здесь мы сохраняем копию изображения чтобы сравнивать ее с оригом но сохраняем в качестве по хужее
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        difference = ImageChops.difference(rgb, recompressed.convert("RGB"))
#Ниже идет сравнение пикселей после сжатия
    stats = ImageStat.Stat(difference)
    mean_difference = sum(stats.mean) / len(stats.mean)
    max_difference = max(channel_max for _, channel_max in difference.getextrema())

    gray = difference.convert("L")
    width, height = gray.size
    columns, rows = min(8, width), min(8, height)
    tile_means = []
    for row in range(rows):
        for column in range(columns):
            box = (
                column * width // columns,
                row * height // rows,
                (column + 1) * width // columns,
                (row + 1) * height // rows,
            )
            tile_means.append(ImageStat.Stat(gray.crop(box)).mean[0])
    tile_stddev = statistics.pstdev(tile_means) if len(tile_means) > 1 else 0.0
    return {
        "mean_difference": mean_difference,
        "max_difference": max_difference,
        "tile_stddev": tile_stddev,
    }


    #Тут мы делим картинку на боксы для более детального анализа
def _local_noise_analysis(image, output_dir, file_stem, grid_size=12):
    rgb = image.convert("RGB")
    rgb_array = np.asarray(rgb)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    #Здесь убирается цвет чтобы в картинке остался шум и детали
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, smooth).astype(np.float32)
    #десь мы режем изображение на ячейки чтобы мы могли читать шум не всего изображения а каждого блока отдельно
    height, width = gray.shape
    columns = min(grid_size, width)
    rows = min(grid_size, height)
    scores = np.zeros((rows, columns), dtype=np.float32)
    tile_values = []

    for row in range(rows):
        for column in range(columns):
            x1 = column * width // columns
            x2 = (column + 1) * width // columns
            y1 = row * height // rows
            y2 = (row + 1) * height // rows
            tile = residual[y1:y2, x1:x2]

            score = float(np.median(tile)) if tile.size else 0.0
            scores[row, column] = score
            tile_values.append(score)

    baseline = float(np.median(scores))
    mad = float(np.median(np.abs(scores - baseline)))
    robust_scale = max(1.4826 * mad, 0.35)
    z_scores = (scores - baseline) / robust_scale


    normalized = np.clip((z_scores + 2.5) / 5.0 * 255, 0, 255).astype(np.uint8)
    enlarged = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(enlarged, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.42, heatmap, 0.58, 0)


    for column in range(1, columns):
        x = column * width // columns
        cv2.line(overlay, (x, 0), (x, height), (235, 235, 235), 1)
    for row in range(1, rows):
        y = row * height // rows
        cv2.line(overlay, (0, y), (width, y), (235, 235, 235), 1)

    map_path = str(Path(output_dir) / f"{file_stem}_noise_map.jpg")
    cv2.imwrite(map_path, overlay)


    unusual = int(np.count_nonzero(np.abs(z_scores) >= 3.0))
    return {
        "map_path": map_path,
        "grid_rows": rows,
        "grid_columns": columns,
        "baseline": baseline,
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "unusual_tiles": unusual,
        "total_tiles": rows * columns,
    }


def _clone_detection(image, output_dir, file_stem):

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    good_matches = []
    displacement = None

    if descriptors is not None and len(keypoints) >= 12:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(descriptors, descriptors, k=3)
        min_distance = max(24.0, 0.025 * float(np.hypot(width, height)))

        for pair in pairs:

            alternatives = [match for match in pair if match.queryIdx != match.trainIdx]
            if len(alternatives) < 2:
                continue
            first, second = alternatives[:2]
            if first.distance < 0.75 * second.distance:
                p1 = np.array(keypoints[first.queryIdx].pt)
                p2 = np.array(keypoints[first.trainIdx].pt)
                if np.linalg.norm(p1 - p2) >= min_distance:
                    good_matches.append(first)

        if len(good_matches) >= 8:
            vectors = []
            for match in good_matches:
                p1 = np.array(keypoints[match.queryIdx].pt)
                p2 = np.array(keypoints[match.trainIdx].pt)
                vectors.append(p2 - p1)
            vectors = np.asarray(vectors)


            quantized = np.round(vectors / 24.0).astype(int)
            keys, counts = np.unique(quantized, axis=0, return_counts=True)
            best_index = int(np.argmax(counts))
            best_count = int(counts[best_index])
            if best_count >= 6:
                target_vector = keys[best_index] * 24.0
                selected = []
                for match, vector in zip(good_matches, vectors):
                    if np.linalg.norm(vector - target_vector) <= 30.0:
                        selected.append(match)
                if len(selected) >= 6:
                    displacement = target_vector.tolist()
                    good_matches = selected
                else:
                    good_matches = []
            else:
                good_matches = []
        else:
            good_matches = []

    if good_matches:
        for match in good_matches:
            p1 = tuple(np.round(keypoints[match.queryIdx].pt).astype(int))
            p2 = tuple(np.round(keypoints[match.trainIdx].pt).astype(int))
            cv2.circle(output, p1, 7, (0, 255, 255), 2)
            cv2.circle(output, p2, 7, (0, 80, 255), 2)
            cv2.line(output, p1, p2, (255, 255, 0), 1)
        status = "Обнаружена группа похожих ключевых точек с согласованным смещением. Возможен copy-move фрагмент."
    else:
        status = "Убедительная группа совпадений ORB с единым смещением не обнаружена. Это не исключает копирование фрагментов."

    map_path = str(Path(output_dir) / f"{file_stem}_clone_map.jpg")
    cv2.imwrite(map_path, output)
    return {
        "map_path": map_path,
        "keypoints": len(keypoints),
        "matches": len(good_matches),
        "displacement": displacement,
        "status": status,
    }


def analyze_image(image_path, output_dir=None):
    path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = path.stem

    with Image.open(path) as source:
        image = source.copy()
        image_format, width, height, mode = source.format or "Неизвестно", *source.size, source.mode
        metadata = _read_metadata(source)
        ela = _ela_analysis(source)

    noise = _local_noise_analysis(image, output_dir, file_stem)
    clone = _clone_detection(image, output_dir, file_stem)

    make = _safe_text(metadata.get("Make"))
    model = _safe_text(metadata.get("Model"))
    date_time = _safe_text(metadata.get("DateTimeOriginal") or metadata.get("DateTime"))
    software = _safe_text(metadata.get("Software"))
    editor_found = any(word in software.lower() for word in EDITOR_KEYWORDS)
    indicators = []

    if editor_found:
        indicators.append(f"В поле Software указано ПО, связанное с редактированием: {software}.")
    if not metadata:
        indicators.append("EXIF-метаданные отсутствуют.")
    elif not metadata.get("Make") and not metadata.get("Model"):
        indicators.append("В EXIF нет сведений о производителе и модели устройства.")

    if ela["mean_difference"] >= 12 or ela["tile_stddev"] >= 7:
        indicators.append("ELA показывает заметную или неравномерную разницу после повторного JPEG-сохранения. Это может быть связано с редактированием, но также встречается из-за сжатия и особенностей изображения.")
    else:
        indicators.append("По используемым порогам выраженных ELA-аномалий не обнаружено. Это не доказывает отсутствие редактирования.")

    if noise["unusual_tiles"]:
        indicators.append(
            f"Локальный анализ шума: {noise['unusual_tiles']} из {noise['total_tiles']} участков заметно отличаются от медианного уровня высокочастотного остатка. Проверьте их по карте шума."
        )
    else:
        indicators.append("Локальный анализ шума не выявил участков, превышающих выбранный порог относительного отклонения.")
    indicators.append(clone["status"])

    gps = metadata.get("GPSInfo") or {}
    gps_coordinates = None
    try:
        if gps.get("GPSLatitude") and gps.get("GPSLongitude"):
            gps_coordinates = (
                _gps_to_degrees(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N")),
                _gps_to_degrees(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E")),
            )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass

    return {
        "file_name": path.name, "format": image_format, "width": width, "height": height,
        "mode": mode, "make": make, "model": model, "date_time": date_time,
        "software": software, "has_exif": bool(metadata), "gps_coordinates": gps_coordinates,
        "editor_found": editor_found, "ela": ela, "noise": noise, "clone": clone,
        "indicators": indicators,
    }


def format_report(report):
    gps = report["gps_coordinates"]
    if gps:
        lat, lon = gps
        gps_text = f"{lat:.6f}, {lon:.6f}\nКарта: https://maps.google.com/?q={lat:.6f},{lon:.6f}"
    else:
        gps_text = "Нет координат или их не удалось прочитать."

    ela = report["ela"]
    noise = report["noise"]
    clone = report["clone"]
    indicators = "\n".join(f"• {item}" for item in report["indicators"])
    displacement = clone["displacement"]
    displacement_text = f"{displacement[0]:.0f}, {displacement[1]:.0f} px" if displacement else "Не определено"

    return (
        "АНАЛИЗ ИЗОБРАЖЕНИЯ\n------------------------------\n"
        f"Файл: {report['file_name']}\nФормат: {report['format']}\n"
        f"Размер: {report['width']} × {report['height']}\nЦветовая модель: {report['mode']}\n\n"
        "EXIF-МЕТАДАННЫЕ\n"
        f"Производитель: {report['make']}\nМодель: {report['model']}\n"
        f"Дата съёмки: {report['date_time']}\nПрограмма в метаданных: {report['software']}\nGPS: {gps_text}\n\n"
        "ELA-АНАЛИЗ\n"
        f"Средняя разница: {ela['mean_difference']:.2f}\n"
        f"Максимальная разница: {ela['max_difference']}\n"
        f"Разброс по участкам: {ela['tile_stddev']:.2f}\n\n"
        "ЛОКАЛЬНЫЙ АНАЛИЗ ШУМА\n"
        f"Сетка: {noise['grid_columns']} × {noise['grid_rows']}\n"
        f"Минимальная оценка участка: {noise['min_score']:.2f}\n"
        f"Максимальная оценка участка: {noise['max_score']:.2f}\n"
        f"Выделяющиеся участки: {noise['unusual_tiles']} из {noise['total_tiles']}\n"
        f"Карта шума: {noise['map_path']}\n\n"
        "ПОИСК КОПИРОВАННЫХ ФРАГМЕНТОВ\n"
        f"Ключевые точки ORB: {clone['keypoints']}\n"
        f"Согласованные совпадения: {clone['matches']}\n"
        f"Оценка смещения: {displacement_text}\n"
        f"Карта совпадений: {clone['map_path']}\n"
        f"{clone['status']}\n\n"
        "ОБЩИЕ НАБЛЮДЕНИЯ\n"
        f"Указание редактора в Software: {'Да' if report['editor_found'] else 'Нет'}\n"
        f"{indicators}\n\n"
        "ВАЖНО: это эвристические методы, а не доказательство подделки или подлинности. "
        "Различия шума могут возникать из-за текстур, освещения, резкости, размытия и сжатия. "
        "Clone Detection может находить повторяющиеся естественные детали и пропускать изменённые, "
        "повёрнутые или масштабированные копии. Проверяйте отмеченные области визуально."
    )
