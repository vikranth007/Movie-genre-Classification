from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load vectorizer and model
with open("Tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("genre_model.pkl", "rb") as f:
    model = pickle.load(f)

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
