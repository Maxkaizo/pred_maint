FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml uv.lock /app/

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Use uv run to execute within the virtual environment
CMD ["/app/entrypoint.sh"]
