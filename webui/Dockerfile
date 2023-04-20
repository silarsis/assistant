FROM node

EXPOSE 3000

WORKDIR /app
COPY client .

# Temporary hack fix for SSL intercept on some machines
RUN npm config set strict-ssl false && npm install

CMD ["/usr/local/bin/npm", "run", "start"]