from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import joblib
import os

app = FastAPI(title="Определитель Сцены")

os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static", html=False), name="static")

try:
    yolo_model = YOLO('model/yolov8n.pt')
    xgb_model = joblib.load('model/best_scene_classifier_xgb_onto.joblib')
except Exception as e:
    yolo_model = None
    xgb_model = None



@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h1>Ошибка: index.html не найден → {e}</h1>"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not xgb_model or not yolo_model:
        raise HTTPException(500, "Модели не загружены")

    temp_path = f"temp_{file.filename}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        results = yolo_model(temp_path, verbose=False)
        detected = {yolo_model.names[int(cls)] for r in results for cls in r.boxes.cls}

        from ontology_utils import prepare_features
        features = prepare_features(detected)

        pred_encoded = int(xgb_model.predict(features)[0])
        confidence = float(xgb_model.predict_proba(features).max())

        class_mapping = {
            0: "home",
            1: "office",
            2: "restaurant",
            3: "store",
            4: "street"
        }

        scene = class_mapping.get(pred_encoded, f"unknown ({pred_encoded})")

        return {
            "scene": scene,
            "confidence": round(confidence * 100, 2),
            "detected_objects": sorted(list(detected))
        }
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
