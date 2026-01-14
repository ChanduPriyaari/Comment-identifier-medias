import joblib
from preprocess import clean_text, is_invalid, contains_abuse, is_negative

model = joblib.load("model/toxic_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_labels(comment: str):
    text = clean_text(comment)

    # 1. Invalid
    if is_invalid(text):
        return ["Invalid"]

    # 2. Toxic / Harassment
    if contains_abuse(text):
        return ["Toxic"]

    # 3. ML Prediction
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    # 4. Negative override
    if is_negative(text):
        return ["Negative"]

    # 5. Safe fallback
    return ["Safe"]
