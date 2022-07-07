FROM python:3.9

WORKDIR /app

ENV PATH="/root/.poetry/bin:${PATH}"

COPY cr_predictor.py pyproject.toml README.md ./
ADD cr_model ./cr_model

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python - && \
    mkdir /app/training_set && \
    cd /app && \
    poetry update && \
    poetry install

ENTRYPOINT ["poetry", "run", "cr_predictor"]
CMD ["--help"]