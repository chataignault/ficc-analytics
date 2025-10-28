
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Install harlequin and curl
RUN uv tool install harlequin
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Download Chinook sample SQLite database
RUN curl -o chinook.db https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite

ENV PATH="/root/.local/bin:$PATH"
CMD ["harlequin", "chinook.db"]

