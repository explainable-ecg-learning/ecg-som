# ECG-SOM Explorer — Vercel Static Deployment

Zero-cold-start, static CDN build of the interactive **ECG-SOM Explorer**
supporting **SOM Explorer**, **Cell Compare**, **UDA Explorer** (168-run
benchmark + precomputed UDA Playground demos), and **System Overview** across
**PTB-XL**, **Georgia (CinC 2020)**, and **Chapman-Shaoxing**.

## Why Static Export?

PyTorch CPU wheels exceed Vercel's `250 MB` serverless bundle limit and add
slow cold starts. Instead, [`explorer/build_vercel.py`](../build_vercel.py)
exports precomputed SOM grids, per-cell embeddings & demographic histograms
(for client-side Cell Compare), lazy per-cell waveforms/percentiles, and
precomputed UDA adapted maps directly into `./data/`.

You can also connect a live FastAPI GPU/CPU backend at any time by appending
`?api=https://your-fastapi-host` to the deployed Vercel URL.

## Deploy to Vercel

### Option 1 — Vercel Dashboard / GitHub Auto-Deploy (Recommended)
1. Push this repository to GitHub.
2. Import the repository in the Vercel Dashboard.
3. Set **Root Directory** to `explorer/vercel`, leave **Framework Preset** as `Other`,
   leave **Build Command** empty, and click **Deploy**.
4. Open the project's **Analytics** tab and click **Enable** to activate Vercel
   Web Analytics (`/_vercel/insights/script.js` is already wired in `index.html`).

### Option 2 — Vercel CLI
```bash
cd explorer/vercel
npx vercel --prod
```

## Regenerating the Static Bundle

Whenever checkpoints, showcase splits, or benchmark results change, regenerate
`explorer/vercel/`:

```bash
python explorer/build_vercel.py
```

And preview locally before deploying:

```bash
python3 -m http.server 8041 --directory explorer/vercel
```
