FROM python:3.12-slim

WORKDIR /app

# git is required at pip-install time: requirements.txt pulls tvdatafeed
# straight from its GitHub repo (no PyPI release exists) — slim base
# images don't ship git, so pip's clone step fails without this.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Serving-only deps — docs/04: no torch/optuna/mlflow on HF, keep the
# image lean for the 2 vCPU / 16GB free-tier box.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces' Docker SDK routes traffic to this port (see README.md's
# `app_port` front-matter, must match).
EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
