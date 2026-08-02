#!/usr/bin/env bash
# Install the bot as a systemd service.
#
#   sudo bash deploy/install.sh
#
# Works on both families Oracle Cloud offers: Ubuntu/Debian (apt) and
# Oracle Linux/RHEL (dnf).  Oracle's default image is Oracle Linux, whose
# login user is `opc` rather than `ubuntu`, so assuming one distribution
# silently breaks half the installs.
#
# Run it from a checkout of this repository.  It never writes secrets: the
# environment file is created empty and you fill it in afterwards.
set -euo pipefail

APP_DIR=/opt/crypto-forecaster
ENV_FILE=/etc/crypto-forecaster.env
SERVICE=crypto-forecaster
BOT_USER=botuser
MINIMUM_PYTHON="3.11"

if [[ $EUID -ne 0 ]]; then
  echo "Bu script root olarak calismalidir: sudo bash deploy/install.sh" >&2
  exit 1
fi

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$SOURCE_DIR/run.py" ]]; then
  echo "run.py bulunamadi; script'i depo kopyasindan calistirin." >&2
  exit 1
fi

# --- Python selection -------------------------------------------------------
# pyproject requires >= 3.11.  Oracle Linux 9 still ships 3.9 as `python3`, so
# prefer an explicitly versioned interpreter and fall back to the default.
python_is_new_enough() {
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null
}

find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_is_new_enough "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

echo "==> Paketler kuruluyor"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv python3-pip git rsync ca-certificates
elif command -v dnf >/dev/null 2>&1; then
  # python3.11 lives in AppStream on Oracle Linux 8 and 9.  `|| true` because
  # a newer image may already ship a good enough default python3.
  dnf install -y -q git rsync ca-certificates
  dnf install -y -q python3.11 python3.11-pip 2>/dev/null || true
  if ! find_python >/dev/null; then
    dnf install -y -q python3.12 python3.12-pip 2>/dev/null || true
  fi
else
  echo "Desteklenmeyen paket yoneticisi; apt-get veya dnf gerekli." >&2
  exit 1
fi

PYTHON_BIN=$(find_python) || {
  echo "Python $MINIMUM_PYTHON+ bulunamadi. Kurun ve script'i tekrar calistirin." >&2
  exit 1
}
echo "    Python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

echo "==> Servis kullanicisi: $BOT_USER"
NOLOGIN=$(command -v nologin || echo /sbin/nologin)
if ! id -u "$BOT_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell "$NOLOGIN" "$BOT_USER"
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
  "$PYTHON_BIN" -m venv "$APP_DIR/.venv"
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
# Oracle Linux ships SELinux enforcing; without the right label systemd cannot
# execute anything under /opt.  Harmless where SELinux is absent.
if command -v restorecon >/dev/null 2>&1; then
  restorecon -R "$APP_DIR" 2>/dev/null || true
fi
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
