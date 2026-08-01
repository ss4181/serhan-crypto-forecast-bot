#!/usr/bin/env bash
# Pull the latest code into the running install and restart the service.
#
#   sudo bash deploy/update.sh [checkout-dizini]
#
# Runtime state (data, artifacts, state) is never touched.
set -euo pipefail

APP_DIR=/opt/crypto-forecaster
SERVICE=crypto-forecaster
BOT_USER=botuser
SOURCE_DIR=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

if [[ $EUID -ne 0 ]]; then
  echo "Bu script root olarak calismalidir: sudo bash deploy/update.sh" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_DIR/run.py" ]]; then
  echo "run.py bulunamadi: $SOURCE_DIR" >&2
  exit 1
fi

echo "==> Testler"
if ! "$APP_DIR/.venv/bin/python" -m unittest discover -s "$SOURCE_DIR/tests" -t "$SOURCE_DIR/tests" -q; then
  echo "Testler basarisiz; guncelleme durduruldu." >&2
  exit 1
fi

echo "==> Kod kopyalaniyor"
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude 'data' \
  --exclude 'artifacts' --exclude 'state' --exclude 'web' \
  "$SOURCE_DIR"/ "$APP_DIR"/
"$APP_DIR/.venv/bin/python" -m pip install --quiet -e "$APP_DIR"
chown -R "$BOT_USER":"$BOT_USER" "$APP_DIR"

echo "==> Servis yeniden baslatiliyor"
systemctl restart "$SERVICE"
sleep 3
systemctl --no-pager --lines=10 status "$SERVICE"
