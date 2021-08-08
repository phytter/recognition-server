from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import recognition.model as model
from pydantic import BaseModel
from typing import List
import uvicorn

class PredictOut(BaseModel):
    label: str
    score: float

    class Config:
        schema_extra = {
            "example": {
                "label": "Pacu",
                "score": 0.98,
            }
        }

class AvailableClassesOut(BaseModel):
    classes: List[str] = []
    total: int

    class Config:
        schema_extra = {
            "example": {
                "classes": ["Pacu", "Tambatinga"],
                "total": 2,
            }
        }

app = FastAPI(title="Reconhecimento de peixes - API")

recognizer = model.Recognizer('/app/recognition/model.h5')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_description="Predição da classe do peixe contido na imagem", response_model=PredictOut)
async def predict_image(file: UploadFile = File(...)):
    file = await file.read() ## byte file
    predict_label, score  = recognizer.predict(file)
    return {"label": predict_label, 'score': score}

@app.get("/available-classes", response_model=AvailableClassesOut)
async def get_available_classes():
    return {"classes": recognizer.get_classes(), "total": len(recognizer.classes)}

@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)