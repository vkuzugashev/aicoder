import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool

# Загрузка переменных окружения
load_dotenv()

# Путь к директории скрипта
SCRIPT_DIR = Path(__file__).resolve().parent.parent
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
@tool
def list_dir(path: str = "") -> str:
    """
    Возвращает список элементов директории с указанием типа: файл или папка.
    Возвращает строку с результатом или сообщение об ошибке.
    """
    dir_path = _safe_path(path, WORK_DIR)
    
    # Проверяем существование директории
    if not os.path.exists(dir_path):
        return f"❌ Директория не найдена: '{path}'"
    
    if not os.path.isdir(dir_path):
        return f"❌ Путь не является директорией: '{path}'"
    
    try:
        items = []
        for name in os.listdir(dir_path):
            item_path = os.path.join(dir_path, name)
            item_type = "📁" if os.path.isdir(item_path) else "📄"
            items.append(f"{item_type} {name}")
        
        if not items:
            return f"📁 Директория '{path}' пуста"
        
        result = f"Содержимое '{path}':\n" + "\n".join(items)
        return result
        
    except PermissionError:
        return f"❌ Нет прав на чтение директории: '{path}'"
    except Exception as e:
        return f"❌ Ошибка при чтении директории: {str(e)}"

@tool
def create_dir(path: str) -> bool:
    """Создает директорию."""
    dir_path = _safe_path(path, WORK_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)
    return True

@tool
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

@tool
# --- Функции для работы с файлами ---
def read_file(path: str, encoding: str = "utf-8") -> str:
    """Читает файл. Если путь относительный, ищет в SOURCE_DIR, иначе в WORK_DIR."""
    # Пытаемся прочитать из исходников (только чтение)
    file_path = _safe_path(path, WORK_DIR)
    mode = "r"
    if (WORK_DIR / 'src').resolve() in file_path.parents:
        encoding="cp1251"
    
    # Проверяем существование файла
    if not os.path.exists(file_path):
        return f"❌ Файл не найден: '{path}'"
    
    if os.path.isdir(file_path):
        return f"❌ Путь является директорией, а не файлом: '{path}'"    
    
    try:
        with open(file_path, mode, encoding=encoding) as f:
            return f.read()
    except Exception as e:
        raise ValueError(f'Error read file: {file_path}, encoding: {encoding}, message: {e}') from e

@tool
def write_file(path: str, content: str, overwrite: bool = False) -> bool:
    """
    Записывает файл в рабочую директорию.
    Если файл существует и overwrite=False — выбрасывает ошибку.
    """
    file_path = _safe_path(path, WORK_DIR)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not overwrite:
        return f"❌ Файл уже существует: '{path}'. Используй overwrite=True для перезаписи"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True

@tool
def create_file(path: str, content: str) -> bool:
    """Создает файл с контентом. Ошибается, если файл уже есть."""
    return write_file(path, content, overwrite=False)  # Явно запрещаем перезапись

@tool
def delete_file(path: str) -> bool:
    """Удаляет файл."""
    file_path = _safe_path(path, WORK_DIR)
    if file_path.exists():
        file_path.unlink()
        return True
    return False

@tool
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

@tool
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

@tool
def pwd():
    """
    Получить путь к текущей рабочей директории
    """
    try:        
        # return os.getcwd()
        return WORK_DIR
    except Exception as e:
        return f"Ошибка: {str(e)}"

