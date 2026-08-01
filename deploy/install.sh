#!/usr/bin/env bash
# Install the bot as a systemd service on a fresh Ubuntu host
# (tested target: Oracle Cloud Always Free, Ubuntu 22.04/24.04, ARM or x86).
#
#   sudo bash deploy/install.sh
#
# Run it from a checkout of this repository.  It never writes secrets: the
# environment file is created empty and you fill it in afterwards.
set -euo pipefail

APP_DIR=/opt/crypto-forecaster
ENV_FILE=/etc/crypto-forecaster.env
SERVICE=crypto-forecaster
BOT_USER=botuser

if [[ $EUID -ne 0 ]]; then
  echo "Bu script root olarak calismalidir: sudo bash deploy/install.sh" >&2
  exit 1
fi

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$SOURCE_DIR/run.py" ]]; then
  echo "run.py bulunamadi; script'i depo kopyasindan calistirin." >&2
  exit 1
fi

echo "==> Paketler kuruluyor"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git rsync ca-certificates

echo "==> Servis kullanicisi: $BOT_USER"
if ! id -u "$BOT_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$BOT_USER"
fi

echo "==> Kod $APP_DIR altina kopyalaniyor"
mkdir -p "$APP_DIR"
# Keep runtime state; replace only the code.
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude 'data' \
  --exclude 'artifacts' --exclude 'state' --exclude 'web' \
  "$SOURCE_DIR"/ "$APP_DIR"/
mkdir -p "$APP_DIR"/{data,artifacts/models,artifacts/reports,state/telegram,state/outcomes}

echo "==> Sanal ortam ve bagimliliklar"
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/python" -m pip install --quiet --upgrade pip
"$APP_DIR/.venv/bin/python" -m pip install --quiet -e "$APP_DIR"

chown -R "$BOT_USER":"$BOT_USER" "$APP_DIR"

echo "==> Ortam dosyasi: $ENV_FILE"
if [[ ! -f "$ENV_FILE" ]]; then
  install -m 600 -o root -g root "$SOURCE_DIR/deploy/env.example" "$ENV_FILE"
  echo "    Olusturuldu (bos). Degerleri doldurmadan servis mesaj gonderemez."
else
  chmod 600 "$ENV_FILE"
  echo "    Mevcut dosya korundu."
fi

echo "==> systemd birimi"
install -m 644 "$SOURCE_DIR/deploy/$SERVICE.service" "/etc/systemd/system/$SERVICE.service"
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null

cat <<EOF

Kurulum tamam.

1) Gizli degerleri girin:
     sudo nano $ENV_FILE
2) Servisi baslatin:
     sudo systemctl restart $SERVICE
3) Ilk tur ~365 gunluk veriyi indirir ve alti modeli arastirir (5-10 dakika).
   Canli gunlugu izleyin:
     sudo journalctl -u $SERVICE -f

Durum:      systemctl status $SERVICE
Guncelleme: sudo bash $APP_DIR/deploy/update.sh   (yeni kodu cektikten sonra)

Unutmayin: GitHub Actions kopyasini standby birakin, yoksa her uyari iki kez gider.
EOF
