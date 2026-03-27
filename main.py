import os
import asyncio
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

# ── API Keys (set these in Render Environment Variables — never hardcode) ──
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
REPLICATE_TOKEN = os.environ.get("REPLICATE_TOKEN", "")

# ── Models ──
HF_IMAGE_MODEL        = "black-forest-labs/FLUX.1-schnell"
HF_API_URL            = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
REPLICATE_VIDEO_MODEL = "minimax/video-01"


# ══════════════════════════════════════════════
#  HELPER — upload to tmpfiles.org
# ══════════════════════════════════════════════

async def upload_to_tmpfiles(data: bytes, filename: str, mime: str) -> str:
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
async def root(
    q: str = Query(None),
    prompt: str = Query(None),
):
    text = q or prompt
    if text:
        return await generate_image(q=text)
    return {
        "name": "BRAINS JET AI Media API",
        "creator": "MR.ROMANTIC",
        "endpoints": {
            "image": "/?q=your image prompt",
            "video": "/video?q=your video prompt",
        }
    }


# ══════════════════════════════════════════════
#  IMAGE  (HuggingFace FLUX via new router)
# ══════════════════════════════════════════════

@app.get("/image")
async def generate_image(
    q: str = Query(None),
    prompt: str = Query(None),
):
    text = q or prompt
    if not text:
        return JSONResponse({"error": "Missing ?q= parameter"}, status_code=400)

    if not HF_TOKEN:
        return JSONResponse({"error": "HF_TOKEN not configured on server"}, status_code=500)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(3):
                r = await client.post(
                    HF_API_URL,
                    headers={
                        "Authorization": f"Bearer {HF_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={"inputs": text},
                )

                if r.status_code == 503:
                    # Model still loading — wait and retry
                    try:
                        wait = r.json().get("estimated_time", 20)
                    except Exception:
                        wait = 20
                    await asyncio.sleep(min(float(wait), 30))
                    continue

                if r.status_code != 200:
                    return JSONResponse(
                        {"error": f"HuggingFace error {r.status_code}: {r.text[:300]}"},
                        status_code=500,
                    )
                break
            else:
                return JSONResponse(
                    {"error": "Model still loading. Please try again in 30 seconds."},
                    status_code=503,
                )

        image_url = await upload_to_tmpfiles(r.content, "image.png", "image/png")

        return JSONResponse({
            "url": image_url,
            "prompt": text,
            "model": HF_IMAGE_MODEL,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ══════════════════════════════════════════════
#  VIDEO  (Replicate minimax)
# ══════════════════════════════════════════════

@app.get("/video")
async def generate_video(
    q: str = Query(None),
    prompt: str = Query(None),
):
    text = q or prompt
    if not text:
        return JSONResponse({"error": "Missing ?q= parameter"}, status_code=400)

    if not REPLICATE_TOKEN:
        return JSONResponse({"error": "REPLICATE_TOKEN not configured on server"}, status_code=500)

    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Step 1 — create prediction
            r = await client.post(
                f"https://api.replicate.com/v1/models/{REPLICATE_VIDEO_MODEL}/predictions",
                headers=headers,
                json={"input": {"prompt": text}},
            )

            if r.status_code not in (200, 201):
                return JSONResponse(
                    {"error": f"Replicate error {r.status_code}: {r.text[:300]}"},
                    status_code=500,
                )

            prediction = r.json()
            prediction_id = prediction["id"]
            status = prediction.get("status", "starting")
            poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
            deadline = time.time() + 240  # 4 minute max

            # Step 2 — poll until done
            while status not in ("succeeded", "failed", "canceled"):
                if time.time() > deadline:
                    return JSONResponse(
                        {"error": "Video generation timed out. Try a shorter prompt."},
                        status_code=504,
                    )
                await asyncio.sleep(5)
                pr = await client.get(poll_url, headers=headers)
                prediction = pr.json()
                status = prediction.get("status", "starting")

            if status != "succeeded":
                err = prediction.get("error", "Generation failed")
                return JSONResponse({"error": str(err)}, status_code=500)

            output = prediction.get("output")
            video_url = output[0] if isinstance(output, list) else output

            if not video_url:
                return JSONResponse({"error": "No video URL in response"}, status_code=500)

            return JSONResponse({
                "url": video_url,
                "prompt": text,
                "model": REPLICATE_VIDEO_MODEL,
            })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
