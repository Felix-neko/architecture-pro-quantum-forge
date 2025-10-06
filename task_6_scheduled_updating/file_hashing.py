import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def calculate_hashes(in_folder_path: Path) -> Dict[Path, str]:
    """
    Вычисляет blake2b хеши для всех файлов в папке (нерекурсивно).

    Args:
        in_folder_path: путь к папке для обработки

    Returns:
        словарь {файл_путь: хеш_строка}

    Raises:
        ValueError: если путь не существует или не является папкой
    """
    # Проверяем, что путь существует и это папка
    if not in_folder_path.exists():
        raise ValueError(f"Путь не существует: {in_folder_path}")

    if not in_folder_path.is_dir():
        raise ValueError(f"Путь не является папкой: {in_folder_path}")

    hashes = {}

    # Нерекурсивный обход: только файлы в корне папки
    for file_path in in_folder_path.iterdir():
        # Пропускаем директории, обрабатываем только файлы
        if not file_path.is_file():
            continue

        # Создаём новый объект blake2b для каждого файла
        h = hashlib.blake2b()

        # Читаем файл и обновляем хеш (работает и для пустых файлов)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):  # Читаем блоками по 8KB
                h.update(chunk)

        # Сохраняем хеш в виде hex-строки
        hashes[file_path] = h.hexdigest()

    return hashes


def compare_hashes(
    old_hashes: Dict[Path, str], new_hashes: Dict[Path, str]
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Функция сравнения двух словарей с хешами файлов.

    Args:
        old_hashes: словарь старых хешей {путь: хеш}
        new_hashes: словарь новых хешей {путь: хеш}

    Returns:
        кортеж из трёх списков: (новые_файлы, изменённые_файлы, удалённые_файлы)
    """
    old_paths = set(old_hashes.keys())
    new_paths = set(new_hashes.keys())

    # Новые файлы: есть в new, но нет в old
    new_files = list(new_paths - old_paths)

    # Удалённые файлы: есть в old, но нет в new
    deleted_files = list(old_paths - new_paths)

    # Изменённые файлы: есть в обоих, но хеши отличаются
    common_paths = old_paths & new_paths
    modified_files = [path for path in common_paths if old_hashes[path] != new_hashes[path]]

    return new_files, modified_files, deleted_files


if __name__ == "__main__":
    old_folder_path = Path(__file__).parent.parent / "task_2_sample_dataset/arcanum_articles/text_output_replaced"
    new_folder_path = (
        Path(__file__).parent.parent / "task_2_sample_dataset/arcanum_articles/text_output_replaced_modified"
    )
    some_temp_folder_path = Path("some_temp_folder")

    shutil.rmtree(some_temp_folder_path, ignore_errors=True)
    shutil.copytree(old_folder_path, some_temp_folder_path, dirs_exist_ok=True)
    old_hashes = calculate_hashes(some_temp_folder_path)

    shutil.rmtree(some_temp_folder_path, ignore_errors=True)
    shutil.copytree(new_folder_path, some_temp_folder_path, dirs_exist_ok=True)
    new_hashes = calculate_hashes(some_temp_folder_path)

    new_files, modified_files, deleted_files = compare_hashes(old_hashes, new_hashes)

    print(f"new_files: {new_files}")
    print(f"modified_files: {modified_files}")
    print(f"deleted_files: {deleted_files}")

    shutil.rmtree(some_temp_folder_path, ignore_errors=True)
