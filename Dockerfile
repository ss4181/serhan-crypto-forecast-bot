# Alternative to the systemd install for hosts that prefer containers.
#
#   docker build -t crypto-forecaster .
#   docker run -d --name crypto-forecaster --restart unless-stopped \
#     --env-file /etc/crypto-forecaster.env \
#     -v crypto-data:/app/data -v crypto-artifacts:/app/artifacts \
#     -v crypto-state:/app/state crypto-forecaster
#
# The volumes matter: delivery receipts live in /app/state and losing them
# means an already sent alert can go out a second time.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CRYPTO_BOT_ROLE=primary

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -e . \
    && useradd --system --create-home botuser \
    && mkdir -p data artifacts/models artifacts/reports state/telegram state/outcomes \
    && chown -R botuser:botuser /app

COPY run.py ./
COPY tests ./tests

USER botuser

# No exposed port and no inbound surface: the bot only makes outbound HTTPS
# calls to Binance market data and the Telegram Bot API.
HEALTHCHECK --interval=5m --timeout=30s --start-period=15m \
    CMD python -c "import pathlib,sys,time; p=pathlib.Path('artifacts/reports/cloud_snapshot.json'); sys.exit(0 if p.exists() and time.time()-p.stat().st_mtime < 900 else 1)"

CMD ["python", "-u", "run.py", "serve", "--days", "365", "--poll-seconds", "60"]
