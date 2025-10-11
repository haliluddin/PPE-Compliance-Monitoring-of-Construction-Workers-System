set -euo pipefail

cd /workspace/ppe-monitor

if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

export POSTGRES_URL="sqlite:///./ppe_local.db"
export REDIS_URL="redis://localhost:6379/0"
export USE_CELERY=false

if ! which redis-server > /dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y redis-server
fi

if ! redis-cli ping > /dev/null 2>&1; then
  redis-server --daemonize yes
  sleep 0.5
fi

if ! redis-cli ping > /dev/null 2>&1; then
  echo "ERROR: redis did not start"
  exit 1
fi

nohup uvicorn app_triton_http:app --host 0.0.0.0 --port 9000 --workers 1 > uvicorn.log 2>&1 &

sleep 0.5
ss -ltnp | grep 9000 || true
tail -n 40 uvicorn.log || true

cd frontend

if ! which npm > /dev/null 2>&1; then
  apt-get update
  curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
  DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
fi

npm ci --silent || true

nohup npm run dev -- --host 0.0.0.0 > vite.log 2>&1 &

sleep 0.5
ss -ltnp | grep 5173 || true
tail -n 40 vite.log || true
