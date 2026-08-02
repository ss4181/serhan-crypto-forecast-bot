#!/usr/bin/env bash
# Pull the latest code into the running install and restart the service.
#
#   sudo bash deploy/update.sh [checkout-dizini]
#
# Runtime state (data, artifacts, state) is never touched.
#
# The tests run against the staged copy, not against the previous install.
# Testing new tests with the old library only ever proves that the library
# changed, which is exactly what an update is for -- an earlier version of this
# script did that and could never pass a release that touched both.
set -euo pipefail

APP_DIR=/opt/crypto-forecaster
BACKUP_DIR=/opt/crypto-forecaster.previous
SERVICE=crypto-forecaster
BOT_USER=botuser
SOURCE_DIR=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

CODE_ONLY=(
  --exclude '.git' --exclude '.venv' --exclude 'data'
  --exclude 'artifacts' --exclude 'state' --exclude 'web'
)

if [[ $EUID -ne 0 ]]; then
  echo "Bu script root olarak calismalidir: sudo bash deploy/update.sh" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_DIR/run.py" ]]; then
  echo "run.py bulunamadi: $SOURCE_DIR" >&2
  exit 1
fi
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  echo "$APP_DIR/.venv bulunamadi; once deploy/install.sh calistirin." >&2
  exit 1
fi

restore() {
  echo "==> Onceki surum geri yukleniyor"
  rsync -a --delete "${CODE_ONLY[@]}" "$BACKUP_DIR"/ "$APP_DIR"/
  "$APP_DIR/.venv/bin/python" -m pip install --quiet -e "$APP_DIR"
  chown -R "$BOT_USER":"$BOT_USER" "$APP_DIR"
}

echo "==> Mevcut surum yedekleniyor"
rm -rf "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
rsync -a "${CODE_ONLY[@]}" "$APP_DIR"/ "$BACKUP_DIR"/

echo "==> Yeni kod hazirlaniyor"
# Safe while the service runs: Python already holds its modules in memory, so
# swapping files only matters at the restart below.
rsync -a --delete "${CODE_ONLY[@]}" "$SOURCE_DIR"/ "$APP_DIR"/
"$APP_DIR/.venv/bin/python" -m pip install --quiet -e "$APP_DIR"

echo "==> Testler (yeni kod)"
if ! "$APP_DIR/.venv/bin/python" -m unittest discover -s "$APP_DIR/tests" -t "$APP_DIR/tests" -q; then
  echo "Testler basarisiz; degisiklik geri alindi, servis dokunulmadan calisiyor." >&2
  restore
  exit 1
fi

chown -R "$BOT_USER":"$BOT_USER" "$APP_DIR"

echo "==> Servis yeniden baslatiliyor"
systemctl restart "$SERVICE"
sleep 3
systemctl --no-pager --lines=10 status "$SERVICE"
