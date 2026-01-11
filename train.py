import joblib
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from data_loader import load_dataset
from preprocess import clean_text
from features import build_vectorizer

LABEL_COLUMNS = [
    "insult",
    "threat",
    "hate",
    "harassment",
    "love",
    "support"
]

def train_model(csv_path: str):
    # Load data
    df = load_dataset(csv_path)

    # Clean text
    df["clean_comment"] = df["comment"].apply(clean_text)

    X = df["clean_comment"]
    y = df[LABEL_COLUMNS]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vectorization
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        )
    )

    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=LABEL_COLUMNS))

    # Save artifacts
    joblib.dump(model, "model/toxic_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

if __name__ == "__main__":
    train_model("data/comments.csv")
