FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /app
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY routecraft ./routecraft
RUN python -m pip install --no-cache-dir .
COPY artifacts ./artifacts
RUN useradd --create-home --uid 10001 router
USER router
EXPOSE 8030
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8030/ready', timeout=3)"
CMD ["python", "-m", "routecraft.cli", "serve", "--host", "0.0.0.0"]
