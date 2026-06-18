# 🎭 RCUBEFACESWAP

> Seamless, high-performance, and API-driven Face Swapping powered by InsightFace and FastAPI.

![Face Swap API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![AI Models](https://img.shields.io/badge/AI-InsightFace-FF4B4B?style=for-the-badge&logo=python&logoColor=white)
![Frontend](https://img.shields.io/badge/Frontend-Vanilla_JS-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

Welcome to **RCUBEFACESWAP** — an elegant and fully functional AI application that allows you to swap faces between a source image and a target image in seconds. This repository serves as a creative showcase of bridging state-of-the-art deep learning (`InsightFace`/`ONNX`) with a blazing-fast backend (`FastAPI`) and a clean, responsive frontend.

## 🚀 Features

- **Blazing Fast API**: Built on top of FastAPI for asynchronous, high-speed request handling.
- **State-of-the-art AI**: Uses the powerful `buffalo_l` FaceAnalysis model and `inswapper_128.onnx` to accurately map, align, and swap facial features.
- **Smart Image Processing**: Automatically resizes massive images to optimize inference time without losing output quality.
- **Seamless Blending**: Employs OpenCV's `seamlessClone` for hyper-realistic face blending, adjusting lighting and skin tones to match the target.
- **Sleek Web Interface**: An out-of-the-box HTML/JS frontend to interact with the API effortlessly.

## 🛠️ Project Architecture

Here is a quick look at what's happening under the hood:
1. **`app/main.py`**: The central nervous system. It initializes the FastAPI server, pre-loads the heavy AI models into memory on startup (to avoid cold starts), and handles the `/swap` endpoint routes.
2. **`app/swap.py`**: The AI core. It detects faces using InsightFace, extracts embeddings, generates masks, and elegantly swaps the source face onto the target body.
3. **`app/utils.py`**: The utility toolbelt. Handles smart image resizing to prevent out-of-memory crashes and ensures outputs are saved in maximum quality.
4. **`index.html`**: A stylish, vanilla HTML/JS frontend that interacts with the backend asynchronously, complete with image previews and dynamic loading states.

## 💻 Installation & Setup

### Requirements
- Python 3.10+
- **ONNX Model**: This project requires the `inswapper_128.onnx` model file to work. Since it is too large for GitHub, you must download it manually.

### 📥 Model Setup
1. Download the `inswapper_128.onnx` model file (you can typically find it on HuggingFace, e.g., [download link](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx)).
2. Create a folder named `models` in the root directory (if it doesn't exist).
3. Place the downloaded `inswapper_128.onnx` file directly inside the `models/` folder.

### Quick Start

1. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Run the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Launch the UI:**
   Open `index.html` in your favorite web browser and start swapping!

## 📡 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Simple health check. Returns `{ "status": "ok" }`. |
| `POST`| `/swap` | Accepts `source` and `target` image files (multipart/form-data). Returns the processed image URL on success. |

## 🎨 Creative Showcase
This repository is designed not just to be functional, but to be a learning tool for anyone looking to integrate heavy AI computer vision models into lightweight, deployable microservices. 

*Built with ❤️ and AI.*
