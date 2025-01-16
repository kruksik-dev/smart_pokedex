FROM python:3.11-slim as builder

RUN pip install --no-cache-dir poetry
WORKDIR /build
COPY pyproject.toml poetry.lock ./
COPY smart_pokedex ./smart_pokedex
RUN poetry build -f wheel



FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /build/dist/*.whl /app/
COPY ./logging.yml /app/
RUN pip install --no-cache-dir /app/*.whl && \
    echo 'export PATH="/usr/local/bin:$PATH"' >> /etc/profile && \
    echo 'export LOGGING_CONFIG_PATH="/app/logging.conf"' >> /etc/profile

ENTRYPOINT ["smart-pokedex"]
CMD ["-i", "/image"]