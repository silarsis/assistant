FROM python:3.9

EXPOSE 8765
EXPOSE 8000

WORKDIR /app

COPY requirements.txt ./

# The next line is for building in a container on a mac
RUN if [ "`uname -m`" = "aarch64" ]; then \
    git clone --recursive https://github.com/nomic-ai/pyllamacpp \
    && cd pyllamacpp \
    && pip install . ; fi
RUN pip install --no-cache-dir -r requirements.txt

ENV HUGGINGFACE_HUB_CACHE /models/
ENV REQUESTS_CA_BUNDLE=

COPY . .

CMD [ "python", "app.py" ]