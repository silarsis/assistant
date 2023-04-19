FROM python:3.9

EXPOSE 8765
EXPOSE 8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV HUGGINGFACE_HUB_CACHE /models/

COPY . .

CMD [ "python", "app.py" ]