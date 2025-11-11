from fastapi import FastAPI, File, UploadFile, HTTPException    
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from DbHandler import SqliteOperationHandler
from skintonedetector import detect_and_crop_face, predict_skin_tone, image_to_base64
from PIL import Image
import io

# APP CONFIGURATION

app = FastAPI(title="AI Skin Tone & Outfit Recommendation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db_handler = SqliteOperationHandler()
db_handler.create_tables()


# REQUEST MODELS

class OutfitColorRequest(BaseModel):
    skintone: str
    outfitcolor: List[str]

class OutfitTypeRequest(BaseModel):
    season: str
    group: str
    outfittype: List[str]

class ProductRequest(BaseModel):
    outfitcolor: str
    outfittype: str
    images: List[str]

# ROUTES

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# ---------- TABLE CREATION ----------
@app.post("/create-tables")
def create_tables():
    db_handler.create_tables()
    return {"message": "All tables created successfully ✅"}


# ---------- OUTFIT COLOR ----------
@app.post("/add-outfit-color")
def add_outfit_color(req: OutfitColorRequest):
    db_handler.AddOutfitColor(req.skintone, req.outfitcolor)
    return {"message": f"✅ Outfit colors added for skintone: {req.skintone}"}

@app.get("/fetch-outfit-color/{skintone}")
def fetch_outfit_color(skintone: str):
    data, status = db_handler.fetchOutFitColor(skintone)
    return {"success": status, "data": data}


# ---------- OUTFIT TYPE ----------
@app.post("/add-outfit-type")
def add_outfit_type(req: OutfitTypeRequest):
    db_handler.AddOutfitType(req.season, req.group, req.outfittype)
    return {"message": f"✅ Outfit types added for {req.season} - {req.group}"}

@app.get("/fetch-outfit-type")
def fetch_outfit_type(season: str, group: str):
    data, status = db_handler.fetchOutfitType(season, group)
    return {"success": status, "data": data}


# ---------- PRODUCT ----------
@app.post("/add-product")
def add_product(req: ProductRequest):
    db_handler.AddProduct(req.outfitcolor, req.outfittype, req.images)
    return {"message": "✅ Product added successfully"}

@app.get("/fetch-product")
def fetch_product(outfitcolor: str, outfittype: str):
    data, status = db_handler.fetchProduct(outfitcolor, outfittype)
    return {"success": status, "data": data}


# ---------- SKIN TONE PREDICTION ----------
@app.post("/predict-skin-tone")
async def predict_skin_tone_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    cropped_face, _ = detect_and_crop_face(image)
    if cropped_face is None:
        raise HTTPException(status_code=404, detail="No face detected in the image")

    pred_class, confidence = predict_skin_tone(cropped_face)
    class_labels = [
        "very_light", "light", "light_medium", "medium", "medium_deep",
        "olive", "tan", "deep", "dark", "very_dark"
    ]

               
    label = class_labels[pred_class]

    face_base64 = image_to_base64(cropped_face)
    return JSONResponse({
        "predicted_class": label,
        "confidence": confidence,
        "cropped_face_base64": face_base64
    })


