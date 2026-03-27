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

# ── API Keys (set in Render Environment Variables — never hardcode) ──
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
FAL_TOKEN = os.environ.get("FAL_TOKEN", "")

# ── Models ──
HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_API_URL     = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
FAL_VIDEO_MODEL = "fal-ai/minimax-video/image-to-video"
FAL_TEXT_VIDEO  = "fal-ai/minimax-video"


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
#  IMAGE  (HuggingFace FLUX)
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
        return JSONResponse({"error": "HF_TOKEN not configured"}, status_code=500)

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
                    {"error": "Model loading. Try again in 30 seconds."},
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
#  VIDEO  (fal.ai minimax)
# ══════════════════════════════════════════════

@app.get("/video")
async def generate_video(
    q: str = Query(None),
    prompt: str = Query(None),
):
    text = q or prompt
    if not text:
        return JSONResponse({"error": "Missing ?q= parameter"}, status_code=400)
    if not FAL_TOKEN:
        return JSONResponse({"error": "FAL_TOKEN not configured"}, status_code=500)

    headers = {
        "Authorization": f"Key {FAL_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:

            # Step 1 — submit job
            r = await client.post(
                f"https://queue.fal.run/{FAL_TEXT_VIDEO}",
                headers=headers,
                json={"prompt": text},
            )

            if r.status_code not in (200, 201):
                return JSONResponse(
                    {"error": f"fal.ai error {r.status_code}: {r.text[:300]}"},
                    status_code=500,
                )

            job = r.json()
            request_id = job.get("request_id")
            if not request_id:
                return JSONResponse(
                    {"error": "No request_id from fal.ai"},
                    status_code=500,
                )

            # Step 2 — poll for result
            status_url = f"https://queue.fal.run/{FAL_TEXT_VIDEO}/requests/{request_id}/status"
            result_url = f"https://queue.fal.run/{FAL_TEXT_VIDEO}/requests/{request_id}"
            deadline = time.time() + 240

            while True:
                if time.time() > deadline:
                    return JSONResponse(
                        {"error": "Video generation timed out. Try a shorter prompt."},
                        status_code=504,
                    )

                await asyncio.sleep(5)
                sr = await client.get(status_url, headers=headers)
                status_data = sr.json()
                status = status_data.get("status", "")

                if status == "COMPLETED":
                    break
                elif status in ("FAILED", "ERROR"):
                    err = status_data.get("error", "Generation failed")
                    return JSONResponse({"error": str(err)}, status_code=500)
                # IN_QUEUE or IN_PROGRESS — keep polling

            # Step 3 — fetch result
            rr = await client.get(result_url, headers=headers)
            result = rr.json()

            # fal.ai returns: {"video": {"url": "..."}} or {"video_url": "..."}
            video_url = (
                result.get("video", {}).get("url")
                or result.get("video_url")
                or result.get("url")
            )

            if not video_url:
                return JSONResponse(
                    {"error": f"No video URL in response: {str(result)[:200]}"},
                    status_code=500,
                )

            return JSONResponse({
                "url": video_url,
                "prompt": text,
                "model": FAL_TEXT_VIDEO,
            })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
