import os
import io
import time
import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Keys (set these in Render Environment Variables) ──
HF_TOKEN     = os.environ.get("HF_TOKEN", "PASTE_YOUR_HF_KEY_HERE")
REPLICATE_TOKEN = os.environ.get("REPLICATE_TOKEN", "PASTE_YOUR_REPLICATE_KEY_HERE")

# ── Models ──
HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
REPLICATE_VIDEO_MODEL = "minimax/video-01"


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

async def upload_to_tmpfiles(data: bytes, filename: str, mime: str) -> str:
    """Upload bytes to tmpfiles.org and return direct download URL."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": (filename, data, mime)},
        )
        result = r.json()
        raw = result["data"]["url"]
        return raw.replace("tmpfiles.org/", "tmpfiles.org/dl/")


# ══════════════════════════════════════════════
#  ROOT
# ══════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "name": "BRAINS JET AI Media API",
        "creator": "MR.ROMANTIC",
        "endpoints": {
            "image": "/?q=your prompt  OR  /image?q=your prompt",
            "video": "/video?q=your prompt",
        }
    }


# ══════════════════════════════════════════════
#  IMAGE GENERATION  (HuggingFace FLUX)
# ══════════════════════════════════════════════

@app.get("/image")
@app.get("/")   # also handle root ?q= like friend's API style
async def generate_image(
    q: str = Query(None, description="Image prompt"),
    prompt: str = Query(None, description="Image prompt (alias)"),
):
    text = q or prompt
    if not text:
        return root()

    hf_url = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # HuggingFace may return 503 while model loads — retry up to 3 times
            for attempt in range(3):
                r = await client.post(
                    hf_url,
                    headers={
                        "Authorization": f"Bearer {HF_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={"inputs": text},
                )
                if r.status_code == 503:
                    # Model loading — wait and retry
                    wait = r.json().get("estimated_time", 20)
                    await asyncio.sleep(min(wait, 30))
                    continue
                if r.status_code != 200:
                    return JSONResponse(
                        {"error": f"HuggingFace error {r.status_code}: {r.text[:200]}"},
                        status_code=500,
                    )
                break
            else:
                return JSONResponse({"error": "Model still loading. Try again in 30 seconds."}, status_code=503)

        image_bytes = r.content
        image_url = await upload_to_tmpfiles(image_bytes, "image.png", "image/png")

        return JSONResponse({
            "url": image_url,
            "prompt": text,
            "model": HF_IMAGE_MODEL,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ══════════════════════════════════════════════
#  VIDEO GENERATION  (Replicate)
# ══════════════════════════════════════════════

@app.get("/video")
async def generate_video(
    q: str = Query(..., description="Video prompt"),
):
    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait=60",
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Step 1 — create prediction
            r = await client.post(
                f"https://api.replicate.com/v1/models/{REPLICATE_VIDEO_MODEL}/predictions",
                headers=headers,
                json={"input": {"prompt": q}},
            )
            if r.status_code not in (200, 201):
                return JSONResponse(
                    {"error": f"Replicate error {r.status_code}: {r.text[:300]}"},
                    status_code=500,
                )

            prediction = r.json()
            prediction_id = prediction["id"]
            status = prediction.get("status", "starting")

            # Step 2 — poll until done (max 4 minutes)
            poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
            deadline = time.time() + 240

            while status not in ("succeeded", "failed", "canceled"):
                if time.time() > deadline:
                    return JSONResponse({"error": "Video generation timed out. Try a shorter prompt."}, status_code=504)
                await asyncio.sleep(5)
                pr = await client.get(poll_url, headers=headers)
                prediction = pr.json()
                status = prediction.get("status", "starting")

            if status != "succeeded":
                err = prediction.get("error", "Generation failed")
                return JSONResponse({"error": str(err)}, status_code=500)

            output = prediction.get("output")
            # output is usually a URL string or list
            if isinstance(output, list):
                video_url = output[0]
            else:
                video_url = output

            if not video_url:
                return JSONResponse({"error": "No video URL in response"}, status_code=500)

            return JSONResponse({
                "url": video_url,
                "prompt": q,
                "model": REPLICATE_VIDEO_MODEL,
            })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── needed for asyncio.sleep inside sync context ──
import asyncio
