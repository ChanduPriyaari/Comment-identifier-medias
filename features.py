from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer():
    """
    Create a TF-IDF vectorizer optimized for toxicity detection.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),       # Capture phrases like "very bad"
        max_features=5000,       # Balance accuracy vs memory
        min_df=1,                 # Ignore extremely rare noise
        max_df=1.0,               # Ignore overly common words
        sublinear_tf=True         # Dampens extreme frequencies
    )

    return vectorizer
