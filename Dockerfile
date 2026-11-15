# Image minimale pour relancer la plateforme à l'identique (ex. serveur de calcul de l'école).
#   docker build -t transilien .
#   docker run --rm -v "$PWD/reports:/app/reports" -v "$PWD/submissions:/app/submissions" \
#     transilien run model=lightgbm_v1
FROM ghcr.io/astral-sh/uv:0.13-python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra explain --no-install-project
COPY . .
RUN uv sync --frozen --no-dev --extra explain

ENTRYPOINT ["uv", "run", "--no-dev", "transilien"]
