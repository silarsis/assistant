FROM python:3.9

EXPOSE 8765
EXPOSE 8000

WORKDIR /app

COPY requirements.txt ./
ENV PYTHONUNBUFFERED 1

# Cloudflare support for SSL intercept :/
# RUN curl -k -o /usr/local/share/ca-certificates/Cloudflare_CA.crt https://developers.cloudflare.com/cloudflare-one/static/documentation/connections/Cloudflare_CA.pem
# RUN dpkg-reconfigure ca-certificates

# The next line is for building in a container on a mac
# Doesn't work yet
RUN if [ "`uname -m`" = "aarch64" ]; then \
    git clone --recursive https://github.com/nomic-ai/pyllamacpp \
    && cd pyllamacpp \
    && pip install . ; fi

RUN /usr/local/bin/python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV HUGGINGFACE_HUB_CACHE /models/

COPY . .

CMD [ "python", "app.py" ]