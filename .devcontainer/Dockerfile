FROM python:3.9

ENV PYTHONUNBUFFERED 1

COPY .. /app
RUN python3 -m pip install --upgrade pip \
    && find /app -iname 'requirements.txt' --exec pip install --no-cache-dir -r {} \;
