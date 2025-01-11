from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 #canny dan histogram
from ultralytics import YOLO #segmentasi
from fastapi.responses import JSONResponse
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = YOLO('C:/KULIAH/SEMESTER 5/pcgk/uas/best.pt') 
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:

        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img) 
        # gambar asli
        buffered_original = BytesIO()
        img.save(buffered_original, format="JPEG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode("utf-8")

        # grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  
        gray_img = Image.fromarray(gray)
        buffered_gray = BytesIO()
        gray_img.save(buffered_gray, format="JPEG")
        gray_base64 = base64.b64encode(buffered_gray.getvalue()).decode("utf-8")

        # canny
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)  # Deteksi tepi
        edges_img = Image.fromarray(edges)
        buffered_edges = BytesIO()
        edges_img.save(buffered_edges, format="JPEG")
        edges_base64 = base64.b64encode(buffered_edges.getvalue()).decode("utf-8")

        # Analisis histogram
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram_img = np.zeros((300, 256, 3), dtype=np.uint8)
        cv2.normalize(histogram, histogram, 0, 300, cv2.NORM_MINMAX)
        for x in range(256):
            cv2.line(histogram_img, (x, 300), (x, 300 - int(histogram[x][0])), (255, 255, 255), 1)
        histogram_pil = Image.fromarray(histogram_img)
        buffered_histogram = BytesIO()
        histogram_pil.save(buffered_histogram, format="JPEG")
        histogram_base64 = base64.b64encode(buffered_histogram.getvalue()).decode("utf-8")

        # segmentasi
        results = model(img)

        # bounding box
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default() 
        predictions = []
        for result in results:
            for box in result.boxes.data.numpy(): 
                x_center, y_center, width, height, conf, cls = box

                x_min = x_center - (width / 12)
                y_min = y_center - (height / 12)
                x_max = x_center + (width / 12)
                y_max = y_center + (height / 12)

                predictions.append({
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max),
                    "confidence": float(conf),
                    "class": int(cls)
                })
                draw.rectangle([x_min, y_min, x_max, y_max], outline="yellow", width=3)

                # Tambahkan label di atas bounding box
                label = "Gigi berlubang"
                text_bbox = draw.textbbox((x_min, y_min), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                draw.rectangle([x_min, y_min - text_height, x_min + text_width, y_min], fill="blue")
                draw.text((x_min, y_min - text_height), label, fill="white", font=font)

        buffered_result = BytesIO()
        img.save(buffered_result, format="JPEG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "predictions": predictions,
            "original_image": original_base64,
            "grayscale_image": gray_base64,
            "edges_image": edges_base64,
            "histogram_image": histogram_base64,
            "segmentation_image": result_base64
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)
