import pandas as pd

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the comment dataset from CSV and validate structure.

    Expected columns:
    - comment
    - insult
    - threat
    - hate
    - harassment
    - love
    - support
    """

    df = pd.read_csv(csv_path)

    required_columns = [
        "comment",
        "insult",
        "threat",
        "hate",
        "harassment",
        "love",
        "support"
    ]

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with empty comments
    df = df.dropna(subset=["comment"])

    return df
