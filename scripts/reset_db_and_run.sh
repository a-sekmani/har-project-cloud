#!/bin/sh
# حذف قاعدة البيانات ثم تشغيل النظام من جديد.
# يشترط: Docker يعمل (افتح Docker Desktop ثم نفّذ هذا السكربت).
set -e
cd "$(dirname "$0")/.."
echo "Stopping containers and removing volumes..."
docker-compose down -v
echo "Starting PostgreSQL..."
docker-compose up -d postgres
echo "Waiting for PostgreSQL to be ready..."
sleep 6
export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://cloudhar:cloudhar@localhost:5433/cloudhar}"
echo "Running migrations..."
./venv/bin/alembic upgrade head
echo "Done. Start the API with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo "Or run with Docker: docker-compose up -d"
if [ "$1" = "--run" ]; then
  exec ./venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
fi
