FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY datatalker/ ./datatalker/
COPY web/ ./web/
RUN python -m datatalker.demo && useradd --create-home app && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn datatalker.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
