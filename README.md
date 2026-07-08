# Semantic Media Search — локальный семантический поиск по изображениям

## О проекте

Desktop-приложение для семантического поиска изображений по текстовому запросу на русском языке. Позволяет индексировать локальную папку с фотографиями и искать их по смысловому описанию, а не по имени файла.

Примеры запросов:
- `фото с морем`
- `машина зимой`
- `люди на концерте`
- `закат в горах`

## Как это работает

```
Папка пользователя → Сканирование изображений → CLIP Embeddings → FAISS Index
                                                                         ↓
Текстовый запрос → Multilingual CLIP → Вектор запроса → FAISS Search → Результаты
```

1. **ImageEncoder** (CLIP ViT-B-32) — строит embedding для каждого изображения
2. **TextEncoder** (multilingual CLIP) — строит embedding для текстового запроса
3. **FAISS** (IndexFlatIP + IndexIDMap2) — находит ближайшие векторы по cosine similarity
4. **SQLite** — хранит метаданные файлов (путь, размер, статус индексации)
5. **PySide6** — графический интерфейс

## Запуск

### Установка

```bash
# Клонировать репозиторий
git clone https://github.com/Vlaov/local-semantic-search.git
cd local-semantic-search

# Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Установить зависимости
pip install -e .
```

### Запуск приложения

```bash
python -m semantic_media_search.main
```

Или через установленную команду:

```bash
semantic-media-search-gui
```

## Сценарий использования

1. Нажать **«Выбрать папку»** и указать папку с изображениями
2. Нажать **«Индексировать»** — прогресс-бар показывает процесс
3. Ввести запрос, например **«фото с морем»**, и нажать **«Найти»**
4. В таблице появятся результаты, отсортированные по релевантности
5. Двойной клик по строке открывает файл

## Структура проекта

```
local-semantic-search/
├── README.md
├── PROJECT_PLAN.md            # Технический план
├── pyproject.toml             # Зависимости и точка входа
├── .gitignore
│
├── src/semantic_media_search/
│   ├── main.py                # Composition root + точка входа
│   ├── config/settings.py     # Конфигурация (пути, имена моделей)
│   ├── domain/models.py       # MediaFile, SearchResult, MediaType
│   ├── scanning/media_scanner.py   # Рекурсивный обход папки
│   ├── ml/
│   │   ├── image_encoder.py        # CLIP image → embedding
│   │   ├── text_encoder.py         # CLIP text → embedding
│   │   └── clip_model.py           # SentenceTransformer wrapper
│   ├── indexing/vector_index.py    # FAISS IndexIDMap2 + save/load
│   ├── storage/
│   │   ├── database.py             # SQLite schema + connect
│   │   └── file_repository.py      # CRUD media_files
│   ├── services/
│   │   ├── indexing_service.py     # Оркестрация индексации
│   │   └── search_service.py       # Оркестрация поиска
│   └── ui/
│       ├── main_window.py          # Главное окно (PySide6)
│       └── indexing_worker.py      # Фоновая индексация (QThread)
│
├── data/                      # Локальные данные (игнорируются git)
│   ├── database/
│   │   └── semantic_media.db
│   └── indexes/
│       └── images.faiss
│
└── tests/                     # Тесты (для будущей разработки)
    ├── unit/
    └── integration/
```

## Технический стек

| Библиотека | Назначение |
|---|---|
| `sentence-transformers` | CLIP-модели для image/text embeddings |
| `torch` | Бэкенд для нейросетей |
| `Pillow` | Загрузка и декодирование изображений |
| `numpy` | Работа с массивами embeddings |
| `faiss-cpu` | Векторный поиск (IndexFlatIP + IndexIDMap2) |
| `PySide6` | Desktop GUI (Qt for Python) |
| `sqlite3` | Метаданные файлов |

## Модели

- **Image encoder**: `sentence-transformers/clip-ViT-B-32`
- **Text encoder**: `sentence-transformers/clip-ViT-B-32-multilingual-v1`

Модели загружаются один раз при старте приложения. Размерность embeddings: 512.

## Разработчики

- **Разработчик A** — ML Core (ImageEncoder, TextEncoder, VectorIndex, SearchService)
- **Разработчик B** — Application/UI Core (FileScanner, SQLite, IndexingService, PySide6 GUI)

## Лицензия

MIT