#!/usr/bin/env bash
# Install a dedicated SSH public key that can only print the sanitized dashboard JSON.
#
#   sudo bash deploy/install-dashboard-key.sh /tmp/trade3-dashboard.pub
set -euo pipefail

APP_DIR=/opt/crypto-forecaster
PUBLIC_KEY_FILE=${1:-}
LOGIN_USER=${2:-${SUDO_USER:-ubuntu}}

if [[ $EUID -ne 0 ]]; then
  echo "Bu script root olarak calismalidir: sudo bash deploy/install-dashboard-key.sh KEY.pub" >&2
  exit 1
fi
if [[ ! -f "$PUBLIC_KEY_FILE" ]]; then
  echo "Dashboard public key dosyasi bulunamadi: $PUBLIC_KEY_FILE" >&2
  exit 1
fi
if ! id "$LOGIN_USER" >/dev/null 2>&1 || [[ "$LOGIN_USER" == root ]]; then
  echo "Gecerli bir root-olmayan SSH kullanicisi gerekli: $LOGIN_USER" >&2
  exit 1
fi
if [[ ! -x "$APP_DIR/.venv/bin/python" || ! -f "$APP_DIR/run.py" ]]; then
  echo "Once trade3 kodunu deploy/update.sh ile guncelleyin." >&2
  exit 1
fi

mapfile -t key_lines < <(sed 's/\r$//' "$PUBLIC_KEY_FILE")
if [[ ${#key_lines[@]} -ne 1 ]]; then
  echo "Public key dosyasi tam olarak bir satir olmali." >&2
  exit 1
fi
public_key=${key_lines[0]}
if [[ ! "$public_key" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+\ trade3-dashboard$ ]]; then
  echo "Beklenen bicim: ssh-ed25519 ANAHTAR trade3-dashboard" >&2
  exit 1
fi

login_home=$(getent passwd "$LOGIN_USER" | cut -d: -f6)
login_group=$(id -gn "$LOGIN_USER")
ssh_dir="$login_home/.ssh"
authorized_keys="$ssh_dir/authorized_keys"
install -d -m 700 -o "$LOGIN_USER" -g "$login_group" "$ssh_dir"
touch "$authorized_keys"
chown "$LOGIN_USER:$login_group" "$authorized_keys"
chmod 600 "$authorized_keys"

if grep -Fq -- "$public_key" "$authorized_keys"; then
  echo "Dashboard public key zaten kurulu."
  exit 0
fi

forced_command='restrict,command="cd /opt/crypto-forecaster && exec /opt/crypto-forecaster/.venv/bin/python /opt/crypto-forecaster/run.py dashboard-export --stdout --source-status fresh"'
printf '%s %s\n' "$forced_command" "$public_key" >> "$authorized_keys"
chown "$LOGIN_USER:$login_group" "$authorized_keys"
chmod 600 "$authorized_keys"

echo "Salt-okunur dashboard anahtari $LOGIN_USER icin kuruldu."
echo "Bu anahtar terminal acamaz; yalniz temizlenmis dashboard JSON'u yazabilir."
