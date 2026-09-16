import os
import io
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

from analyzers.metadata_agent import MetadataAgent
from analyzers.c2pa_agent import C2PAAgent
from analyzers.ela_agent import ELAAgent
from analyzers.noise_fft_agent import NoiseFFTAgent
from analyzers.deep_feature_agent import DeepFeatureAgent
from analyzers.judge_agent import JudgeAgent

# Enforce security decompression limits on Pillow to prevent DecompressionBomb errors
Image.MAX_IMAGE_PIXELS = 89478485

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB max file upload size cap
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(
    title="AI Image Forensics & Multi-Agent Analyzer",
    description="Multi-Agent AI vs Real Image Classification & Forensic Evidence Suite",
    version="1.0.0"
)

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate Agents
metadata_agent = MetadataAgent()
c2pa_agent = C2PAAgent()
ela_agent = ELAAgent()
noise_fft_agent = NoiseFFTAgent()
deep_feature_agent = DeepFeatureAgent()
judge_agent = JudgeAgent()

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_image.png"
    ext = os.path.splitext(filename.lower())[1]

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed upload size of {MAX_FILE_SIZE // (1024*1024)}MB."
        )

    try:
        # Validate that bytes form a legitimate image
        with Image.open(io.BytesIO(image_bytes)) as test_img:
            test_img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is corrupted or not a valid image.")

    # Execute all 5 analyzer agents concurrently in parallel
    results = await asyncio.gather(
        metadata_agent.analyze(image_bytes, filename),
        c2pa_agent.analyze(image_bytes, filename),
        ela_agent.analyze(image_bytes, filename),
        noise_fft_agent.analyze(image_bytes, filename),
        deep_feature_agent.analyze(image_bytes, filename)
    )

    # Synthesize findings with Judge Agent
    verdict_report = judge_agent.synthesize(results)
    verdict_report["filename"] = filename
    verdict_report["file_size_kb"] = round(len(image_bytes) / 1024.0, 1)

    return verdict_report

# Mount static folder and root route
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Image Forensics API is running. Upload to /api/analyze"}
