# 🚀 IPL Prediction Engine: Live Hosting Guide

Follow these steps to host your model on Hugging Face and connect it to the internet for automatic live predictions.

## Step 1: Initialize External Databases
Since Hugging Face storage is ephemeral, you **must** use these external (free) services:
1.  **PostgreSQL (Persistence)**:
    - Create a free account on [Neon.tech](https://neon.tech/) or [Supabase](https://supabase.com/).
    - Copy the **Connection String** (something like `postgres://user:password@host/dbname`).
2.  **Redis (Real-time Streams)**:
    - Create a free account on [Upstash Redis](https://upstash.com/).
    - Copy the **Host**, **Port**, and **Password**.

## Step 2: Initialize Hugging Face Model
We need to upload your local model weights (`models/`) to the Hub so they persist.
1.  Go to your Hugging Face space or model repo: [zeroday01/predictionsingle](https://huggingface.co/zeroday01/predictionsingle).
2.  Get your **HF Access Token** (Write permission) from [Settings > Access Tokens](https://huggingface.co/settings/tokens).
3.  **Run the push script locally**:
    ```bash
    export HF_TOKEN='your_token_here'
    python init_huggingface.py
    ```
    *(This uploads your `hybrid_ensemble_*.pth` files to the permanent Model Hub)*

## Step 3: Create Hugging Face Space (Backend)
1.  Click **"Create New Space"** on Hugging Face.
2.  **SDK**: Choose **Docker**.
3.  **Template**: Blank.
4.  **Hardware**: The default CPU is fine (the model is highly efficient).

## Step 4: Configure Secrets & Variables
In your Hugging Face Space, go to **Settings > Variables and Secrets** and add:

### 🔐 Secrets (Secure)
- `HF_TOKEN`: (Your HF Write Token)
- `DATABASE_URL`: (The Postgres string from Neon/Supabase)
- `REDIS_PASSWORD`: (From Upstash)

### ⚙️ Variables (Public)
- `REDIS_HOST`: (From Upstash)
- `REDIS_PORT`: (e.g. `6379`)
- `HF_REPO_ID`: `zeroday01/predictionsingle`
- `ALLOWED_ORIGINS`: (URL of your frontend or `*`)

## Step 5: Deploy & Monitor
1.  Connect your Git repo or upload the `Dockerfile` and all backend files.
2.  The `Dockerfile` is already configured to run `api_server.py`.
3.  Check the **Log** tab:
    - It should say: `✅ ML Engine successfully loaded from zeroday01/predictionsingle`.
    - It should say: `✅ Connected to PostgreSQL successfully`.
    - It should say: `🚀 Starting Automated Match Discovery Service`.

## Step 6: Keep it Awake
Free Hugging Face spaces sleep after 48 hours. 
1.  Go to [cron-job.org](https://cron-job.org/).
2.  Schedule a job to ping your `/health` endpoint (e.g. `https://your-space-url.hf.space/health`) every 12 hours.

---

### 🎉 Your engine is now LIVE!
- **Auto-Discovery**: It will automatically find the next IPL match on ESPNcricinfo/Cricbuzz.
- **Predictions**: Win probabilities will stream to your dashboard and be logged in your Postgres DB.
- **Auto-Update**: Every 24 hours, the `daily_update_pipeline.py` will retrain the model and push new weights back to the Hub.
