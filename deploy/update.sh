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
BACKUP_READY=0
DEPLOY_SUCCESS=0

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

install_release() {
  if [[ -f "$APP_DIR/requirements.lock" ]]; then
    "$APP_DIR/.venv/bin/python" -m pip install --quiet -r "$APP_DIR/requirements.lock"
    "$APP_DIR/.venv/bin/python" -m pip install --quiet --no-deps --no-build-isolation -e "$APP_DIR"
  else
    # Backward-compatible rollback for the pre-lock release.
    "$APP_DIR/.venv/bin/python" -m pip install --quiet -e "$APP_DIR"
  fi
}

restore() {
  echo "==> Onceki surum geri yukleniyor"
  rsync -a --delete "${CODE_ONLY[@]}" "$BACKUP_DIR"/ "$APP_DIR"/
  install_release
  chown -R "$BOT_USER":"$BOT_USER" "$APP_DIR"
  systemctl restart "$SERVICE"
}

finish_update() {
  local exit_code=$?
  trap - EXIT
  if [[ $exit_code -ne 0 && $BACKUP_READY -eq 1 && $DEPLOY_SUCCESS -eq 0 ]]; then
    if ! restore; then
      echo "GERI DONUS BASARISIZ: onceki surum yeniden yuklenemedi." >&2
    fi
  fi
  exit "$exit_code"
}

echo "==> Mevcut surum yedekleniyor"
rm -rf "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
rsync -a "${CODE_ONLY[@]}" "$APP_DIR"/ "$BACKUP_DIR"/
BACKUP_READY=1
trap finish_update EXIT

echo "==> Yeni kod hazirlaniyor"
# Safe while the service runs: Python already holds its modules in memory, so
# swapping files only matters at the restart below.
rsync -a --delete "${CODE_ONLY[@]}" "$SOURCE_DIR"/ "$APP_DIR"/
install_release

echo "==> Testler (yeni kod)"
if ! "$APP_DIR/.venv/bin/python" -m unittest discover -s "$APP_DIR/tests" -t "$APP_DIR/tests" -q; then
  echo "Testler basarisiz; onceki surume geri donuluyor." >&2
  exit 1
fi

chown -R "$BOT_USER":"$BOT_USER" "$APP_DIR"
# Keep the installed unit in sync with the checked-out release.  Without this
# explicit install, changing deploy/crypto-forecaster.service would only update
# the copy under /opt and systemd would continue running the old command.
install -m 644 "$SOURCE_DIR/deploy/crypto-forecaster.service" \
  "/etc/systemd/system/crypto-forecaster.service"
# Membership state contains Telegram identifiers.  Older releases may have
# created it with 0755/0644 defaults, so every update repairs those permissions
# before the service starts.
install -d -m 700 -o "$BOT_USER" -g "$BOT_USER" "$APP_DIR/state/telegram"
for private_state in members.json pending_members.json; do
  if [[ -f "$APP_DIR/state/telegram/$private_state" ]]; then
    chmod 600 "$APP_DIR/state/telegram/$private_state"
  fi
done

echo "==> Servis yeniden baslatiliyor"
systemctl daemon-reload
systemctl restart "$SERVICE"
sleep 3
systemctl --no-pager --lines=10 status "$SERVICE"
DEPLOY_SUCCESS=1
