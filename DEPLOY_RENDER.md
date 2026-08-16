# Deploying Sehnsucht to Render.com (free, no Gradio, no card required)

## 1. Put these files in a GitHub repo
```
main.py
requirements.txt
static/index.html
model.pth          <- your trained model
tokenizer.model    <- your SentencePiece tokenizer
```
Push it to a new public or private GitHub repo. (model.pth is ~200MB —
GitHub's hard file limit is 100MB without Git LFS, so if you hit that,
enable Git LFS: `git lfs install && git lfs track "*.pth"` before adding it.)

## 2. Create the Render account and service
- Go to render.com → sign up (no credit card needed for the free tier)
- New → **Web Service**
- Connect your GitHub repo
- Environment: **Python 3**
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Instance type: **Free**

## 3. Deploy
Render builds and deploys automatically. First build can take a few
minutes (installing torch). You'll get a live URL like:
```
https://sehnsucht.onrender.com
```

## 4. Test it
- Visit the URL in a browser — you'll see the chat UI from `static/index.html`
- Or hit the API directly:
```bash
curl -X POST https://sehnsucht.onrender.com/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 100, "temperature": 0.7}'
```

## 5. What "free" actually means here
- Spins down after ~15 minutes of no traffic, wakes on the next request
  (30–60s cold start — show a loading message, which `index.html` already does)
- 512MB RAM, 0.1 CPU on the free instance. Your model should fit, but
  it's tight. If you hit out-of-memory errors on deploy or first request:
  - Uncomment the `quantize_dynamic` line in `main.py` — shrinks the model's
    memory footprint with minimal code change
  - Confirm `requirements.txt` is pulling the CPU-only torch wheel (it is,
    via the `--extra-index-url` line) — the CUDA build is much larger and
    would likely blow the RAM/disk budget
- 750 free hours/month total — more than enough for a personal project

## 6. APK
No change from the plan already in place — the APK calls
`https://sehnsucht.onrender.com/generate` the same way it would have
called a Hugging Face Space's endpoint. Provider-agnostic by design.
