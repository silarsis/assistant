FROM python:3.9

EXPOSE 8766
EXPOSE 8000

WORKDIR /app

COPY requirements.txt ./
ENV PYTHONUNBUFFERED 1

RUN /usr/local/bin/python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV HUGGINGFACE_HUB_CACHE /models/

COPY . .

CMD [ "python", "app.py" ]
