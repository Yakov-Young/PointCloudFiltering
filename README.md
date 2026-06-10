# PointCloudFiltering

Настольное приложение для фильтрации облаков точек с последующей оценкой с графическим интерфейсом.
Поддерживает несколько алгоритмов фильтрации выбросов.

Оценка происходит по полю бинарной маски выбросов 'scalar_isGarbage' PLY-файла.
'scalar_isGarbage': 
0 - 'норма';
1 - 'выброс'.

## Требования

| Компонент | Версия |
|-----------|--------|
| ОС | Windows 10 |
| Python | 3.12 |
| Пакеты | см. `./requirements.txt` |

## Установка и запуск

Все команды выполняются в **PowerShell**.

**1. Создать и активировать виртуальное окружение:**
```powershell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

**2. Установить зависимости:**
```powershell
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

**3. Запустить приложение:**
```powershell
python .\main.py
```

## Поддерживаемые форматы файлов

Загрузка и сохранение облаков точек: **PLY**

## Фильтры

- **SOR** — Statistical Outlier Removal
- **DSOR** — Dynamic Statistical Outlier Removal  
- **LOF** — Local Outlier Factor
- **Radius Outlier Removal** — удаление по радиусу
- **PCA Curvature Filter** — фильтр на основе кривизны поверхности