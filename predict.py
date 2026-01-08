import json
import pickle
import random


# load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# load intents file
with open("data/intents.json", "r") as f:
    intents = json.load(f)


def predict_intent(text, threshold=0.25):
    text = text.lower()
    X = vectorizer.transform([text])

    probs = model.predict_proba(X)[0]
    classes = model.classes_

    best_index = probs.argmax()
    best_prob = probs[best_index]
    best_intent = classes[best_index]

    if best_prob < threshold:
        return None

    return best_intent


def get_response(intent_tag):
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


def main():
    print("Chatbot is running")
    print("Type 'quit' to stop\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Bot: Bye!")
            break

        intent = predict_intent(user_input)

        if intent is None:
            print("Bot: Sorry, I didn't understand that.")
        else:
            response = get_response(intent)
            print("Bot:", response)


if __name__ == "__main__":
    main()
