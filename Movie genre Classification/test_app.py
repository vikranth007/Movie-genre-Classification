import requests

url = "http://127.0.0.1:8000/predict"

inputs = [
    "A thrilling spy adventure filled with action and suspense.",
    "A heartwarming romantic comedy about two friends finding love.",
    "A terrifying horror story about a haunted house in the woods.",
    "An epic fantasy tale with dragons, wizards, and a quest to save the kingdom.",
    "A documentary exploring the lives of endangered animals in the wild.",
    "A sci-fi journey through space and time to save humanity.",
    "A dramatic story about family struggles and personal growth.",
    "A fun animated movie about talking animals on a wild adventure.",
    "A crime thriller where a detective tries to solve a mysterious murder.",
    "A musical filled with catchy songs and dance numbers."
]







for desc in inputs:
    response = requests.post(url, json={"description": desc})
    print(f"Description: {desc}\nPredicted genre: {response.json()['genre']}\n")
