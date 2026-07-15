FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

RUN useradd --create-home --shell /bin/bash botuser \
    && mkdir -p /app/data \
    && chown -R botuser:botuser /app

COPY --chown=botuser:botuser bot.py /app/bot.py

USER botuser

CMD ["python", "bot.py"]
