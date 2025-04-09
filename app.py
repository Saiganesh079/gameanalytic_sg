from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

RAWG_API_KEY = os.getenv("RAWG_API_KEY")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAWG API Key
API_KEY = os.getenv("RAWG_API_KEY") or "your_rawg_api_key"
BASE_URL = "https://api.rawg.io/api"

# 1. Fetch Genres
@app.get("/genres")
def get_genres():
    response = requests.get(f"{BASE_URL}/genres?key={API_KEY}").json()
    return response.get("results", [])

# 2. Fetch Platforms
@app.get("/platforms")
def get_platforms():
    response = requests.get(f"{BASE_URL}/platforms?key={API_KEY}").json()
    return response.get("results", [])

# 3. Fetch Games with Filters
@app.get("/games")
def get_games(
    genres: Optional[List[str]] = Query(None),
    platform: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    url = f"{BASE_URL}/games?key={API_KEY}&ordering=-rating&page_size=50"
    if genres:
        url += f"&genres={','.join(genres)}"
    if platform:
        url += f"&platforms={platform}"
    if start_date and end_date:
        url += f"&dates={start_date},{end_date}"
    response = requests.get(url).json()
    return response.get("results", [])

# 4. Fetch Companies (Publishers + Developers)
@app.get("/companies")
def get_companies():
    publishers = requests.get(f"{BASE_URL}/publishers?key={API_KEY}").json().get("results", [])
    developers = requests.get(f"{BASE_URL}/developers?key={API_KEY}").json().get("results", [])
    all_companies = {c["name"] for c in publishers + developers}
    return sorted(list(all_companies))

# 5. Game Popularity Prediction
class PredictionInput(BaseModel):
    company: str
    platform_users: int
    genre_popularity: int
    critic_reviews: int
    average_rating: float
    budget: int
    playtime: int
    metacritic_score: int

@app.post("/predict_popularity")
def predict_popularity(data: PredictionInput):
    # Dummy dataset creation
    np.random.seed(42)
    data_size = 500
    dummy_df = pd.DataFrame({
        "company": np.random.choice(["Company A", "Company B", "Company C", data.company], data_size),
        "platform_users": np.random.randint(10, 500, data_size),
        "genre_popularity": np.random.randint(1, 10, data_size),
        "critic_reviews": np.random.randint(50, 5000, data_size),
        "average_rating": np.random.uniform(0, 5, data_size),
        "budget": np.random.randint(1, 500, data_size),
        "playtime": np.random.randint(1, 200, data_size),
        "metacritic_score": np.random.randint(0, 100, data_size),
        "popularity_score": np.random.randint(10, 100, data_size)
    })

    # Encode
    encoder = LabelEncoder()
    dummy_df["company"] = encoder.fit_transform(dummy_df["company"])
    
    scaler = StandardScaler()
    features = ["platform_users", "genre_popularity", "critic_reviews", "average_rating",
                "budget", "playtime", "metacritic_score"]
    dummy_df[features] = scaler.fit_transform(dummy_df[features])

    X = dummy_df.drop("popularity_score", axis=1)
    y = dummy_df["popularity_score"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict
    user_input = pd.DataFrame([{
        "company": data.company,
        **data.dict(exclude={"company"})
    }])
    user_input["company"] = encoder.transform([data.company])
    user_input[features] = scaler.transform(user_input[features])

    prediction = model.predict(user_input)[0]
    importance = model.feature_importances_

    return {
        "predicted_popularity": round(float(prediction), 2),
        "feature_importance": dict(zip(X.columns, map(float, importance)))
    }

# Local run
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
