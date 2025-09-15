
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Install harlequin
RUN uv tool install harlequin

# Copy existing local database
COPY "house_prices.db" .

# Run harlequin
ENV PATH="/root/.local/bin:$PATH"
CMD ["harlequin", "house_prices.db"]

