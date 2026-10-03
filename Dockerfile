FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md openenv.yaml LICENSE ./
COPY shoreline ./shoreline
RUN pip install --no-cache-dir ".[openenv]"
EXPOSE 8000
CMD ["python", "-m", "shoreline.server.app", "--host", "0.0.0.0", "--port", "8000"]
