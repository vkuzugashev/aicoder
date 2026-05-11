import subprocess

from langchain_core.tools import tool

from tools.file_tools import WORK_DIR, _safe_path


@tool
def npm_install(path: str, options: str = ''):
    """
    Выполнение установки пакетов, для вызова передать путь к папке с проектом и опции установки
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


@tool
def npm_build(path: str, options: str = ''):
    """
    Построение проекта. 
    Параметры:
    - path: путь к папке с проектом (например, 'api' или 'client')
    - options: дополнительные опции сборки
    """
    try:
        full_path = _safe_path(path, WORK_DIR)
        
        # Проверяем наличие package.json
        package_json = full_path / 'package.json'
        if not package_json.exists():
            return f"❌ package.json не найден в {path}"
        
        # Безопасная команда
        command = ['npm', 'run', 'build']
        if options:
            command.extend(options.split())
        
        print(f"🖥️ Выполняется: {' '.join(command)}")
        print(f"   Папка: {full_path}")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(full_path),
            shell=False  # ✅ Безопасно
        )
        
        # ✅ Проверяем код возврата
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return f"✅ Сборка успешна:\n{output}"
            return "✅ Сборка успешна (нет вывода)"
        else:
            # ❌ Ошибка сборки
            error_msg = result.stderr.strip() or result.stdout.strip() or "Неизвестная ошибка"
            return f"❌ Ошибка сборки (код {result.returncode}):\n{error_msg}"
            
    except subprocess.TimeoutExpired:
        return "❌ Сборка превысила лимит времени (>180 сек)"
    
    except FileNotFoundError:
        return f"❌ npm не установлен или не найден в PATH"
    
    except PermissionError as e:
        return f"❌ Нет доступа к папке {path}: {e}"
    
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"