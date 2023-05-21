FROM python:3.9

EXPOSE 10000
EXPOSE 8000

WORKDIR /app

COPY requirements.txt ./
ENV PYTHONUNBUFFERED 1

# Cloudflare support for SSL intercept :/
# RUN curl -k -o /usr/local/share/ca-certificates/Cloudflare_CA.crt https://developers.cloudflare.com/cloudflare-one/static/documentation/connections/Cloudflare_CA.pem
# RUN dpkg-reconfigure ca-certificates

RUN /usr/local/bin/python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV HUGGINGFACE_HUB_CACHE /models/

COPY . .

CMD [ "python", "app.py" ]
