from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Step 1: Import karein
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

# Load model and encoder
model = joblib.load("campaign_success_model_final.pkl")
encoder = joblib.load("category_encoder_final.pkl")

app = FastAPI()

# --- Step 2: Yahan CORS middleware add karein ---
origins = [
    "https://fundify.up.railway.app", # Aapka production frontend
    "http://localhost",
    "http://localhost:3000", # Common local development ports
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Sirf in origins se requests allow hongi
    allow_credentials=True,
    allow_methods=["*"], # Sab methods (GET, POST, etc.) allow karein
    allow_headers=["*"], 
)
# -------------------------------------------------

VALID_CATEGORIES = [
    "Creative Arts",
    "Business & Entrepreneurship",
    "Media & Entertainment",
    "Education & Publishing",
    "Lifestyle",
    "Technology & Innovation",
    "Non profit",
    "Medical"
]

class CampaignInput(BaseModel):
    goalAmount: float = Field(..., gt=4999, description="Goal amount in PKR, must be >= 5000")
    category: str
    duration: int = Field(..., gt=0, description="Duration in days, must be > 0")

@app.post("/predict")
def predict_success(data: CampaignInput):
    try:
        if data.category not in VALID_CATEGORIES:
            return {"error": f"Invalid category: {data.category}. Must be one of: {', '.join(VALID_CATEGORIES)}"}

        category_encoded = encoder.transform([data.category])[0]

        X = pd.DataFrame([{
            "goal_pkr": data.goalAmount,
            "category_encoded": category_encoded,
            "launch_to_deadline_days": data.duration
        }])

        success_prob = model.predict_proba(X)[0][1] * 100

        return {"success_probability": round(success_prob, 2)}

    except Exception as e:
        return {"error": str(e)}
