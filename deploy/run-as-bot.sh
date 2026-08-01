#!/usr/bin/env bash
# Run a one-off bot command with the service's own identity and environment.
#
#   sudo bash /opt/crypto-forecaster/deploy/run-as-bot.sh verify-models --send
#   sudo bash /opt/crypto-forecaster/deploy/run-as-bot.sh scorecard --days 7
#   sudo bash /opt/crypto-forecaster/deploy/run-as-bot.sh members --add 123 --name "Ayse"
#
# The environment file is root-only on purpose, so this script reads it and
# drops to the service user instead of loosening the file's permissions.
set -euo pipefail

APP_DIR=/opt/crypto-forecaster
ENV_FILE=/etc/crypto-forecaster.env
BOT_USER=botuser

if [[ $EUID -ne 0 ]]; then
  echo "Bu script root olarak calismalidir: sudo bash $0 <komut>" >&2
  exit 1
fi
if [[ ! -r "$ENV_FILE" ]]; then
  echo "$ENV_FILE bulunamadi; once deploy/install.sh calistirin." >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  echo "Kullanim: sudo bash $0 <komut> [secenekler]" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

cd "$APP_DIR"
exec sudo -u "$BOT_USER" --preserve-env "$APP_DIR/.venv/bin/python" -u run.py "$@"
