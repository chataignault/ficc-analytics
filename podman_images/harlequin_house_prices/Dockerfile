
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Install harlequin 
RUN uv tool install harlequin

# Copy house prices database
COPY house_prices.db .

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"

# Run Harlequin on test database
CMD ["harlequin", "house_prices.db"]

