from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS to allow HTML page to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = joblib.load("Tfidf_vectorizer.pkl")
model = joblib.load("genre_model.pkl")

class InputText(BaseModel):
    description: str

@app.post("/predict")
def predict(input: InputText):
    vec = vectorizer.transform([input.description])
    pred = model.predict(vec)[0]
    return {"genre": pred}
