import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/comments.csv")

# Clean text
df["comment"] = df["comment"].astype(str)
df["cleaned"] = df["comment"].apply(clean_text)

# Labels (force numeric)
label_cols = ["insult", "hate", "threat", "harassment", "love", "support"]
df[label_cols] = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

X = df["cleaned"]
y = df[label_cols]

# Vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9
)

X_vec = vectorizer.fit_transform(X)

# Model
model = MultiOutputClassifier(
    LogisticRegression(max_iter=1000)
)

# Train
model.fit(X_vec, y)

# Save
joblib.dump(model, "model/toxic_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model trained successfully")
