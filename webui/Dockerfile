FROM node

EXPOSE 3000

WORKDIR /app
COPY client .

RUN npm install
CMD ["/usr/local/bin/npm", "run", "start"]