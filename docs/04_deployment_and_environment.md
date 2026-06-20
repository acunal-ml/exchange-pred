# Deployment and Local Environment Strategy

## Local Development (Linux Workspace)
- **Environment:** Linux with a Fish shell; code run interactively (e.g. `Shift+Enter` in VS Code).
- **Service orchestration:** Redis runs locally via daemon or Docker during development.
- **GPU:** RTX 3050 (4GB VRAM) for training only — see MLOps doc for the memory budget.

---

## Production Deployment (Hugging Face Spaces)

**Platform — corrected free-tier facts (verified):**
- **2 vCPU, 16GB RAM, 50GB disk.** CPU Basic is the default free tier.
- **Disk is NON-persistent** — wiped on every rebuild/restart. Nothing written at runtime survives.
- The Space **sleeps after 48h of inactivity**; first request after sleep pays a cold-start penalty (warm-up the model on startup).
- Free **ZeroGPU (H200)** exists but is **quota-limited and ephemeral** — fine for occasional heavy demos, *not* a substitute for persistent low-latency serving. Plan around CPU inference.

**The two contradictions in the original spec, and their fixes:**

1. **"Must cache in Redis" vs. HF reality.** There is **no managed Redis on the free tier.** Resolve via the abstract `CacheBackend` (see data doc): use an in-process `TTLCache`/`diskcache` on HF, or attach an external managed Redis (e.g. Upstash free tier) through a secret. Do not assume a local Redis exists in production.

2. **SQLite persistence vs. ephemeral disk.** A runtime-written SQLite file is lost on rebuild. Options, in order of preference:
   - Seed/rebuild the DB at startup from an **HF Dataset repo**, or
   - Use an external serverless DB (**Turso/libSQL**, Supabase) for anything that must persist, or
   - Accept the cache+API path as the live source and treat SQLite as a warm read-cache only.

**Constraint management:**
- Heavy DL models (LSTM) are **pre-trained locally**; the Space performs **inference only**.
- Serve the **ONNX / int8-quantized** champion via `onnxruntime` (set thread count explicitly for 2 vCPU). Quantize if CPU latency is too high.
- **Model & artifact delivery:** pull the champion (and its scaler/calibrator) from the **HF Hub / Dataset repo** at startup via git-LFS — don't bloat the Space repo or commit large binaries carelessly.
- **Secrets:** API keys, external Redis/DB URLs go in **HF Spaces Secrets**, never in a committed `.env`. `.env.example` documents the keys only.
- **Concurrency:** use the Gradio **queue** (or Streamlit caching) so inference doesn't block the main thread; cache results per `(asset, timeframe)` so concurrent users hit the cache, not the API.

## Dependency split (was missing — this matters a lot on a 2-vCPU box)
- `requirements.txt` (**HF / serving**): lightweight — `gradio`/`streamlit`, `onnxruntime`, `lightgbm` runtime, `pandas`, `numpy`, cache lib, data-source clients. **No `torch`, no `optuna`, no `mlflow`.**
- `requirements-train.txt` (**local**): the heavy stack — `torch`, `optuna`, `mlflow`, `mlflow`, TA-Lib, etc.

Shipping training dependencies to HF wastes build time and RAM and risks OOM on cold start.
