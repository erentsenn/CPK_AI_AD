FROM python:3.11-slim

# Не создавать .pyc и не буферизовать stdout/stderr (удобнее логи в контейнере)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/nltk_data \
    MODEL_PATH=/app/models/baseline_model.pkl

WORKDIR /app

# Устанавливаем зависимости отдельным слоем для кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Заранее скачиваем стоп-слова NLTK, чтобы контейнеру не нужна была сеть в рантайме
RUN python -m nltk.downloader -d /usr/local/nltk_data stopwords

# Копируем код приложения и сериализованную модель
COPY app ./app
COPY models ./models

EXPOSE 8000

# Проверка здоровья контейнера через эндпоинт /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
