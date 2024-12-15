FROM ollama/ollama:latest

COPY ./run-ollama.sh /tmp/run-ollama.sh

WORKDIR /tmp

RUN chmod +x run-ollama.sh \
    && ./run-ollama.sh

CMD [ "ollama", "serve" ]

