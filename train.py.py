import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)        # remove punctuation
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # goood -> good
    return text.strip()



def load_data():
    with open("data/intents.json", "r") as f:
        return json.load(f)


def prepare_data(data):
    texts = []
    labels = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(intent["tag"])

    return texts, labels

def train_model(texts, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, vectorizer


if __name__ == "__main__":
    print("training started")

    data = load_data()
    texts, labels = prepare_data(data)

    model, vectorizer = train_model(texts, labels)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("model trained and saved")
