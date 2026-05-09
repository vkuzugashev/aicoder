import os
from pathlib import Path
import subprocess
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Путь к директории скрипта
SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR / "workdir"

from pathlib import Path
import os

def _safe_path(path: str, base_dir: Path) -> Path:
    """
    Проверяет, что путь находится внутри разрешенной директории.
    
    Args:
        path: Путь к файлу/папке (относительный или абсолютный)
        base_dir: Базовая разрешенная директория
    
    Returns:
        Абсолютный нормализованный путь
    
    Raises:
        PermissionError: Если путь выходит за пределы base_dir
    """
    base_resolved = base_dir.resolve()
    
    # Обработка пустого пути
    if not path or path == "":
        return base_resolved
    
    # Объединяем с base_dir только если путь не абсолютный
    # Иначе используем путь как есть (но всё равно проверяем)
    if os.path.isabs(path):
        full_path = Path(path).resolve()
    else:
        full_path = (base_dir / path).resolve()

    # Проверяем, что путь внутри разрешенной директории
    try:
        full_path.relative_to(base_resolved)
        return full_path
    except ValueError:
        raise PermissionError(
            f"Доступ запрещен!\n"
            f"  Разрешенная директория: {base_resolved}\n"
            f"  Запрошенный путь: {full_path}"
        )

# --- Функции для работы с каталогами ---
def list_dir(path: str = "") -> list:
    """
    Возвращает список элементов директории с указанием типа: файл или папка.
    Пример результата:
    [
        {"name": "src",       "dir": true},
        {"name": "main.py",   "dir": false},
        {"name": "config.json", "dir": false}
    ]
    """
    dir_path = _safe_path(path, WORK_DIR)
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Директория не найдена: {path}")
    
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Путь не является директорией: {path}")

    items = []
    for name in os.listdir(dir_path):
        item_path = os.path.join(dir_path, name)
        item_type = True if os.path.isdir(item_path) else False
        items.append({
            "name": name,
            "dir": item_type
        })
    
    return items

def create_dir(path: str) -> bool:
    """Создает директорию."""
    dir_path = _safe_path(path, WORK_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)
    return True

def delete_dir(path: str) -> bool:
    """Удаляет директорию."""
    dir_path = _safe_path(path, WORK_DIR)
    if not dir_path.exists():
        return False
    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()
        else:
            delete_dir(str(item.relative_to(WORK_DIR)))
    dir_path.rmdir()
    return True

# --- Функции для работы с файлами ---
def read_file(path: str, encoding: str = "utf-8") -> str:
    """Читает файл. Если путь относительный, ищет в SOURCE_DIR, иначе в WORK_DIR."""
    # Пытаемся прочитать из исходников (только чтение)
    file_path = _safe_path(path, WORK_DIR)
    mode = "r"
    if (WORK_DIR / 'src').resolve() in file_path.parents:
        encoding="cp1251"
    try:
        with open(file_path, mode, encoding=encoding) as f:
            return f.read()
    except Exception as e:
        raise ValueError(f'Error read file: {file_path}, encoding: {encoding}, message: {e}') from e

def write_file(path: str, content: str, overwrite: bool = False) -> bool:
    """
    Записывает файл в рабочую директорию.
    Если файл существует и overwrite=False — выбрасывает ошибку.
    """
    file_path = _safe_path(path, WORK_DIR)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not overwrite:
        raise FileExistsError(f"Файл уже существует: {path}. Используй overwrite=True для перезаписи.")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True

def create_file(path: str, content: str) -> bool:
    """Создает файл с контентом. Ошибается, если файл уже есть."""
    return write_file(path, content, overwrite=False)  # Явно запрещаем перезапись

def delete_file(path: str) -> bool:
    """Удаляет файл."""
    file_path = _safe_path(path, WORK_DIR)
    if file_path.exists():
        file_path.unlink()
        return True
    return False

# --- Функции проверки существования ---

def file_exists(path: str) -> bool:
    """
    Проверяет, существует ли файл.
    Возвращает True, если файл существует и это именно файл (не папка).
    """
    try:
        file_path = _safe_path(path, WORK_DIR)
        return file_path.is_file()
    except (PermissionError, ValueError):
        return False


def dir_exists(path: str) -> bool:
    """
    Проверяет, существует ли директория.
    Возвращает True, если путь указывает на существующую папку.
    """
    try:
        dir_path = _safe_path(path, WORK_DIR)
        return dir_path.is_dir()
    except (PermissionError, ValueError):
        return False

def npm_install(path: str, options: str):
    """
    Выполнение установки пакетов
    """
    try:
        full_path = _safe_path(path, WORK_DIR)
        command = f'npm install {options}'
        print(f"🖥️ Выполняется: {command}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=full_path
        )
        output = result.stdout if result.stdout else result.stderr if result.stderr else "✅ Выполнено (нет вывода)"
        return output
    except subprocess.TimeoutExpired:
        return "❌ Команда выполнялась слишком долго (>120 сек)"
    except Exception as e:
        return f"Ошибка: {str(e)}"

def npm_build(path: str, options: str):
    """
    Построение проекта
    """
    try:
        full_path = _safe_path(path, WORK_DIR)
        command = f'npm build {options}'
        print(f"🖥️ Выполняется: {command}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=180, cwd=full_path
        )
        output = result.stdout if result.stdout else result.stderr if result.stderr else "✅ Выполнено (нет вывода)"
        return output
    except subprocess.TimeoutExpired:
        return "❌ Команда выполнялась слишком долго (>180 сек)"
    except Exception as e:
        return f"Ошибка: {str(e)}"

# def chdir(path: str) -> str:
#     """Реально меняет текущую директорию процесса"""
#     try:
#         full_path = _safe_path(path, WORK_DIR)
#         os.chdir(full_path)
#         new_cwd = os.getcwd()
#         print(f"📁 Директория изменена на: {new_cwd}")
#         return f"✅ Текущая директория изменена на: {new_cwd}"
#     except Exception as e:
#         return f"❌ Ошибка при смене директории: {str(e)}"


def pwd():
    """
    Получить путь к текущей рабочей директории
    """
    try:        
        return os.getcwd()
    except Exception as e:
        return f"Ошибка: {str(e)}"