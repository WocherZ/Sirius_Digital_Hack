# Базовый образ для Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию для проекта
WORKDIR /app

# Копируем файл requirements.txt в контейнер
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в контейнер
COPY . .

# Указываем переменную окружения для Streamlit
ENV STREAMLIT_PORT=8501
ENV FASTAPI_PORT=8000

# Открываем порты для Streamlit и FastAPI
EXPOSE $STREAMLIT_PORT
EXPOSE $FASTAPI_PORT

# Указываем команду для запуска двух приложений
CMD ["sh", "-c", "streamlit run app/app.py --server.port $STREAMLIT_PORT & uvicorn backend.main:app --host 0.0.0.0 --port $FASTAPI_PORT"]
