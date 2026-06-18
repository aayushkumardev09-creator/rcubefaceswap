import logging
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.swap import FaceSwapError, swap_faces, _get_face_analyzer, _get_swapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT_DIR / "uploads"
OUTPUT_DIR = ROOT_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.on_event("startup")
def startup_event():
    logger.info("Loading models...")
    # Load models at startup
    analyzer = _get_face_analyzer()
    inswapper = _get_swapper(ROOT_DIR / "models" / "inswapper_128.onnx")
    
    # Warm-up run
    logger.info("Performing warm-up inference...")
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = analyzer.get(dummy_image)
    logger.info("Models loaded and warm-up completed")


@app.get("/health")
def health_check():
    return {"status": "ok"}


async def _save_upload(upload_file: UploadFile, prefix: str) -> Path:
    if not upload_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded files must be images")

    extension = Path(upload_file.filename).suffix or ".jpg"
    filename = f"{prefix}_{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / filename

    content = await upload_file.read()
    destination.write_bytes(content)
    await upload_file.close()
    return destination


@app.post("/swap")
async def swap_endpoint(
    source: UploadFile = File(...), target: UploadFile = File(...)
):
    logger.info("Swap request received")
    source_path = await _save_upload(source, "source")
    target_path = await _save_upload(target, "target")
    output_path = OUTPUT_DIR / f"swap_{uuid4().hex}.png"

    try:
        logger.info("Processing started")
        swap_faces(str(source_path), str(target_path), str(output_path))
        logger.info("Processing completed")
        return JSONResponse(content={
            "success": True,
            "image_url": f"/outputs/{output_path.name}"
        })
    except FaceSwapError as exc:
        logger.error(f"Face swap error: {exc}")
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": str(exc)
        })
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Internal server error"
        })

