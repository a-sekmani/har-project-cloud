FROM python:3.11-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Copy application code (needed for editable install)
COPY app/ ./app/

# Copy Alembic configuration
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Run the application (migrations will be run by docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
