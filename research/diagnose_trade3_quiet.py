"""Read-only Trade3 diagnostics; print only strategy settings and aggregate logs."""

import collections
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

print("SERVER_UTC", datetime.now(timezone.utc).isoformat())
logs = subprocess.check_output(
    ["journalctl", "-u", "crypto-forecaster", "--since", "7 days ago", "--no-pager", "-o", "cat"],
    text=True, errors="replace",
)
scans = []
reasons = collections.Counter()
deliveries = collections.Counter()
regimes = collections.Counter()
long_scans = collections.Counter()
eligible = filters = 0
commands = []
for line in logs.splitlines():
    match = re.search(r"Scalp gozlem: (\d+)/(\d+) taze, rejim (\w+), (\d+) radar, (\d+) kurulum", line)
    if match:
        fresh, attempted, regime, radar, setups = match.groups()
        scans.append((int(fresh), int(attempted), int(radar), int(setups)))
        regimes[regime] += 1
    match = re.search(r"Scalp bildirim filtresi: (\d+) uygun coin/mum; elenme nedenleri: (.*)", line)
    if match:
        filters += 1
        eligible += int(match[1])
        for key, count in re.findall(r"([a-z_]+)=(\d+)", match[2]):
            reasons[key] += int(count)
    match = re.search(r"Scalp Telegram: (.*)", line)
    if match:
        deliveries[match[1]] += 1
    match = re.search(r"LONG Radar: (\d+) piyasa, (\d+) aday, (\w+)", line)
    if match:
        long_scans[(match[3], int(match[2]))] += 1
    if "Komut:" in line:
        commands.append(line)
print("SEVEN_DAY_LOG_SUMMARY", json.dumps({
    "scans": len(scans),
    "scans_with_radar": sum(row[2] > 0 for row in scans),
    "scans_with_setup": sum(row[3] > 0 for row in scans),
    "coin_bar_setup_occurrences": sum(row[3] for row in scans),
    "regimes": dict(regimes), "filter_checks": filters,
    "eligible_occurrences": eligible, "rejection_counts": dict(reasons),
    "scalp_delivery_statuses": dict(deliveries),
    "long_scan_states": {str(key): count for key, count in long_scans.items()},
    "recent_commands": commands[-5:],
}, ensure_ascii=False))

keys = set("""
CRYPTO_BOT_ROLE CRYPTO_MARKET CRYPTO_SCALP_OBSERVATION
CRYPTO_SCALP_MIN_ALERT_SCORE CRYPTO_SCALP_MIN_EXPECTED_NET_BPS
CRYPTO_SCALP_MIN_DIRECTION_PROBABILITY CRYPTO_SCALP_MIN_CALIBRATION_SAMPLES
CRYPTO_SCALP_TRANSITION_ALERTS CRYPTO_SCALP_TRANSITION_MIN_ALERT_SCORE
CRYPTO_SCALP_TRANSITION_MIN_EXPECTED_NET_BPS CRYPTO_SCALP_TRANSITION_MIN_DIRECTION_PROBABILITY
CRYPTO_SCALP_TRANSITION_MIN_CALIBRATION_SAMPLES CRYPTO_SCALP_OFF_ALERTS
CRYPTO_SCALP_ALERT_KILL_SWITCH CRYPTO_SCALP_LIVE_FAMILIES
CRYPTO_LONG_SCOUT CRYPTO_LONG_SCOUT_OWNER_ALERTS CRYPTO_LONG_SCOUT_MIN_SCORE
CRYPTO_LONG_SCOUT_UNIVERSE_LIMIT CRYPTO_SCALP_MIN_QUALITY_PERCENTILE
CRYPTO_SCALP_TRANSITION_MIN_QUALITY_PERCENTILE
""".split())
values = {}
for line in Path("/etc/crypto-forecaster.env").read_text().splitlines():
    key, _, value = line.partition("=")
    if key.strip() in keys:
        values[key.strip()] = value.strip()
print("SELECTED_CONFIG", json.dumps(values))

base = Path("/opt/crypto-forecaster/state/scalp")
shadow_root = base / "experiments/strategy-horizon-shadow-v1"
shadow_path = shadow_root / "summary.json"
if shadow_path.exists():
    shadow = json.loads(shadow_path.read_text())
    print("HORIZON_SHADOW", json.dumps(shadow, ensure_ascii=False))
    profile_id = shadow.get("profileId", "")
    if re.fullmatch(r"[a-f0-9]{16}", profile_id):
        config_path = shadow_root / f"profile-{profile_id}" / "config.json"
        if config_path.exists():
            print("EFFECTIVE_FILTERS", config_path.read_text())
path = base / "notification_status.json"
if path.exists():
    data = json.loads(path.read_text())
    print("CURRENT_NOTIFICATION_STATUS", json.dumps(data, ensure_ascii=False))
path = base / "experiments/long-discovery-v1/state.json"
if path.exists():
    data = json.loads(path.read_text())
    records = data.get("records", [])
    print("LONG_STATE", json.dumps({
        "scannedAtMs": data.get("scannedAtMs"), "regime": data.get("regime"),
        "record_count": len(records),
        "stage_counts": dict(collections.Counter(str(row.get("stage")) for row in records)),
        "owner_alerted_records": sum(bool(row.get("ownerAlert")) for row in records),
        "delivered_candidates": [{
            "symbol": row.get("symbol"), "strategy": row.get("strategy"),
            "sourceCloseMs": row.get("sourceCloseMs"),
            "deliveredAtMs": row["ownerAlert"].get("deliveredAtMs"),
        } for row in records if row.get("ownerAlert")],
    }, ensure_ascii=False))
