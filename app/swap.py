import logging
import os
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from app.utils import resize_if_too_large


logger = logging.getLogger(__name__)


class FaceSwapError(RuntimeError):
    pass


_FACE_ANALYZER: FaceAnalysis | None = None
_SWAPPER = None


def _get_face_analyzer(model_name: str = "buffalo_l") -> FaceAnalysis:
    global _FACE_ANALYZER
    if _FACE_ANALYZER is None:
        print("Initializing InsightFace model...")
        analyzer = FaceAnalysis(name=model_name)
        analyzer.prepare(ctx_id=-1, det_size=(640, 640))  # Force CPU
        _FACE_ANALYZER = analyzer
        print("InsightFace model loaded")
    return _FACE_ANALYZER


def _get_swapper(model_path: str | Path):
    global _SWAPPER
    if _SWAPPER is None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"inswapper ONNX model not found at {model_path}")
        _SWAPPER = get_model(str(model_path))
    return _SWAPPER


def _read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image from {path}")
    return image


def _detect_faces(face_analyzer: FaceAnalysis, image: np.ndarray) -> list:
    return face_analyzer.get(image)


def _align_face(image: np.ndarray, face) -> np.ndarray:
    return norm_crop(image, face.kps, image_size=128)


def _preprocess_face(face_image: np.ndarray) -> np.ndarray:
    image = face_image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None, ...]
    return image


def _postprocess_output(output: np.ndarray) -> np.ndarray:
    image = output[0]
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    return image


def _swap_face(source_embedding: np.ndarray, target_face: np.ndarray, session: onnxruntime.InferenceSession) -> np.ndarray:
    target_input = _preprocess_face(target_face)
    feed = {'source': source_embedding, 'target': target_input}
    outputs = session.run(None, feed)
    if not outputs:
        raise FaceSwapError("ONNX model did not return any outputs")
    return _postprocess_output(outputs[0])


def _create_face_mask(shape: tuple[int, int], bbox: np.ndarray) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _blend_face(target_image: np.ndarray, swapped_face: np.ndarray, face) -> np.ndarray:
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(target_image.shape[1], x2), min(target_image.shape[0], y2)
    if x1 >= x2 or y1 >= y2:
        raise FaceSwapError("Invalid target face bounding box")

    face_width = x2 - x1
    face_height = y2 - y1
    resized_face = cv2.resize(swapped_face, (face_width, face_height), interpolation=cv2.INTER_LINEAR)
    output_image = target_image.copy()
    mask = _create_face_mask(output_image.shape[:2], np.array([x1, y1, x2, y2], dtype=np.int32))
    try:
        center = tuple(map(int, np.mean(face.kps, axis=0)))
        output_image = cv2.seamlessClone(resized_face, output_image, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        output_image[y1:y2, x1:x2] = resized_face
    return output_image


def swap_faces(source_path: str, target_path: str, output_path: str) -> None:
    try:
        face_analyzer = _get_face_analyzer()
        swapper = _get_swapper(Path(__file__).resolve().parents[1] / "models" / "inswapper_128.onnx")

        print("Loading source image")
        source_img = cv2.imread(source_path)
        if source_img is None:
            raise FaceSwapError(f"Unable to read source image from {source_path}")
        source_img = resize_if_too_large(source_img)

        print("Loading target image")
        target_img = cv2.imread(target_path)
        if target_img is None:
            raise FaceSwapError(f"Unable to read target image from {target_path}")
        target_img = resize_if_too_large(target_img)

        print("Detecting faces in source")
        source_faces = face_analyzer.get(source_img)
        if not source_faces:
            print("Error: No face detected in source image")
            raise FaceSwapError("No face detected in source image")

        print("Detecting faces in target")
        target_faces = face_analyzer.get(target_img)
        if not target_faces:
            print("Error: No face detected in target image")
            raise FaceSwapError("No face detected in target image")

        # Sort by detection score descending
        source_faces = sorted(source_faces, key=lambda x: x.det_score, reverse=True)
        target_faces = sorted(target_faces, key=lambda x: x.det_score, reverse=True)

        print(f"Source faces found: {len(source_faces)}")
        print(f"Target faces found: {len(target_faces)}")
        print(f"Source detection scores: {[f'{face.det_score:.3f}' for face in source_faces]}")
        print(f"Target detection scores: {[f'{face.det_score:.3f}' for face in target_faces]}")

        source_face = source_faces[0]  # highest score
        result = target_img.copy()

        print("Starting swap")
        for face in target_faces:
            result = swapper.get(result, face, source_face, paste_back=True)

        print("Saving output")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, result):
            raise FaceSwapError(f"Unable to write output image to {output_path}")
        
        if os.path.exists(output_path):
            print(f"Output saved at {output_path}")
            print("Swap test successful")
        else:
            print(f"Error: Output file not found at {output_path}")
            raise FaceSwapError(f"Output file not found at {output_path}")
    except Exception as e:
        print(f"Swap failed with error: {e}")
        raise

