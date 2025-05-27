import joblib
from fastapi import FastAPI
from pydantic   import BaseModel


model = joblib.load("genre_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class MovieInput(BaseModel):
    description : str


app = FastAPI()


@app.post("/predict")

def predict_genre(data: MovieInput):
    text = [data.description.lower()]
    text_vec = vectorizer.transform(text)
    prediction = model.predict(text_vec)[0]
    print(f"Input: {text[0]}")
    print(f"Prediction: {prediction}")
    return {"genre": prediction}

if __name__ == "__main__":
    app.run(debug=True)
