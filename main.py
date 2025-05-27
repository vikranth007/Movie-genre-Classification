from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # ✅ use joblib instead of pickle

app = FastAPI()

# Load vectorizer and model
vectorizer = joblib.load("Tfidf_vectorizer.pkl")
model = joblib.load("genre_model.pkl")

class InputText(BaseModel):
    description: str

@app.get("/")
def read_root():
    return {"message": "Movie Genre Classifier is running."}

@app.post("/predict")
def predict_genre(input: InputText):
    text_vector = vectorizer.transform([input.description])
    prediction = model.predict(text_vector)
    return {"genre": prediction[0]}
