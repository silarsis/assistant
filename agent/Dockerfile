FROM python:3.11

EXPOSE 7860

WORKDIR /app

COPY requirements.txt ./
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y portaudio19-dev python3-pyaudio

RUN /usr/local/bin/python -m pip install --root-user-action=ignore --upgrade pip \
    && pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

ENV HUGGINGFACE_HUB_CACHE /models/

COPY *.py .
COPY character.txt .
COPY models/ ./models/
# COPY .vscode/launch.json .vscode/launch.json

CMD [ "python", "client_gradio.py" ]
