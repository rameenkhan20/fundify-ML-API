from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

# Load model and encoder
model = joblib.load("campaign_success_model_final.pkl")
encoder = joblib.load("category_encoder_final.pkl")

app = FastAPI()

# ✅ Add valid categories list here
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
    goal_pkr: float = Field(..., gt=4999, description="Goal amount in PKR, must be >= 5000")
    category: str
    launch_to_deadline_days: int = Field(..., gt=0, description="Duration in days, must be > 0")

@app.post("/predict")
def predict_success(data: CampaignInput):
    try:
        # ✅ Validate category here
        if data.category not in VALID_CATEGORIES:
            return {"error": f"Invalid category: {data.category}. Must be one of: {', '.join(VALID_CATEGORIES)}"}

        category_encoded = encoder.transform([data.category])[0]

        X = pd.DataFrame([{
            "goal_pkr": data.goal_pkr,
            "category_encoded": category_encoded,
            "launch_to_deadline_days": data.launch_to_deadline_days
        }])

        success_prob = model.predict_proba(X)[0][1] * 100

        return {"success_probability": round(success_prob, 2)}

    except Exception as e:
        return {"error": str(e)}



# from fastapi import FastAPI
# from pydantic import BaseModel, Field
# import joblib
# import numpy as np
# import pandas as pd

# # Load model and encoder
# model = joblib.load("campaign_success_model_final.pkl")
# encoder = joblib.load("category_encoder_final.pkl")

# app = FastAPI()

# # Add valid categories list here
# VALID_CATEGORIES = [
#     "Creative Arts",
#     "Business & Entrepreneurship",
#     "Media & Entertainment",
#     "Education & Publishing",
#     "Lifestyle",
#     "Technology & Innovation",
#     "Non profit",
#     "Medical"
# ]

# class CampaignInput(BaseModel):
#     goal_pkr: float = Field(..., gt=4999, description="Goal amount in PKR, must be >= 5000")
#     category: str
#     launch_to_deadline_days: int = Field(..., gt=0, description="Duration in days, must be > 0")

# @app.post("/predict")
# def predict_success(data: CampaignInput):
#     try:
#         # ✅ Validate category here
#         if data.category not in VALID_CATEGORIES:
#             return {"error": f"Invalid category: {data.category}. Must be one of: {', '.join(VALID_CATEGORIES)}"}

#         category_encoded = encoder.transform([data.category])[0]

#         X = pd.DataFrame([{
#             "goal_pkr": data.goal_pkr,
#             "category_encoded": category_encoded,
#             "launch_to_deadline_days": data.launch_to_deadline_days
#         }])

#         success_prob = model.predict_proba(X)[0][1] * 100

#         return {"success_probability": round(success_prob, 2)}

#     except Exception as e:
#         return {"error": str(e)}
