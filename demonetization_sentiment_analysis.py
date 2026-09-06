#!/usr/bin/env python3
"""
Demonetization Twitter Sentiment Analysis & Narrative Mining
=============================================================
Clean, modern, production-ready version for GitHub / HR submission.

Compatible with:
- Python 3.9+
- gensim 4.x
- pandas 2.x
- scikit-learn 1.x

Author: Project Submission Ready
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import ssl
from collections import Counter

# Fix SSL certificate issues on macOS (common with NLTK downloads)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from textblob import TextBlob
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CSV_PATH          = "demonetization-tweets.csv"
SAMPLE_SIZE       = 6000          # Set to None to use full dataset
VECTOR_SIZE       = 80
N_CLUSTERS_RANGE  = range(4, 9)
RANDOM_STATE      = 42

# ------------------------------------------------------------------
# 0. Ensure NLTK resources
# ------------------------------------------------------------------
def ensure_nltk():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            print(f"   Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)

print("Checking NLTK resources...")
ensure_nltk()

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
def load_data(path=CSV_PATH, sample=SAMPLE_SIZE):
    print("=" * 60)
    print("1. Loading data...")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{path}' not found.\n"
            "Please place demonetization-tweets.csv in the same folder as this script."
        )
    df = pd.read_csv(path, encoding="ISO-8859-1")
    print(f"   Original shape : {df.shape}")
    df = pd.DataFrame({"tweet": df["text"].astype(str)})
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"   Sampled to     : {df.shape}")
    return df

# ------------------------------------------------------------------
# 2. Cleaning
# ------------------------------------------------------------------
def clean_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"^rt\s*@\w+:?", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"<ed>|<U\+[\w]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\b\w{1,2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str, stop_words: set) -> str:
    tokens = word_tokenize(text)
    return " ".join(w for w in tokens if w not in stop_words and len(w) > 2)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("2. Cleaning tweets...")
    df["cleaned_tweet"] = df["tweet"].apply(clean_tweet)

    stop_words = set(stopwords.words("english"))
    extra = {
        "rt", "amp", "via", "https", "http", "co",
        "demonetization", "demonetisation", "modi", "narendra", "india"
    }
    stop_words.update(extra)

    df["fully_cleaned_tweet"] = df["cleaned_tweet"].apply(
        lambda x: remove_stopwords(x, stop_words)
    )
    df = df[df["fully_cleaned_tweet"].str.len() > 5].reset_index(drop=True)
    print(f"   After cleaning : {df.shape}")
    return df

# ------------------------------------------------------------------
# 3. Sentiment Analysis
# ------------------------------------------------------------------
def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print("3. Computing sentiment (TextBlob)...")
    df["polarity"] = df["fully_cleaned_tweet"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    def label(p):
        if p > 0.05:
            return 1   # Positive
        if p < -0.05:
            return 2   # Negative
        return 3       # Neutral

    df["sentiment"] = df["polarity"].apply(label)
    print("   Sentiment distribution:")
    print(df["sentiment"].value_counts().sort_index().to_string())
    return df

# ------------------------------------------------------------------
# 4. Word2Vec Embeddings + Feature Matrix
# ------------------------------------------------------------------
def build_features(df: pd.DataFrame):
    print("4. Training Word2Vec & building feature matrix...")
    df["tokenized"] = df["fully_cleaned_tweet"].apply(word_tokenize)

    model = Word2Vec(
        sentences=df["tokenized"].tolist(),
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=3,
        workers=4,
        epochs=8,
        seed=RANDOM_STATE
    )

    def sent_vec(tokens):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        if not vecs:
            return np.zeros(VECTOR_SIZE, dtype=np.float32)
        return np.mean(vecs, axis=0)

    vectors = np.vstack(df["tokenized"].apply(sent_vec).values)
    print(f"   Vectors shape  : {vectors.shape}")

    scaler = MinMaxScaler()
    vectors = scaler.fit_transform(vectors)

    # Append scaled polarity as extra feature
    pol = ((df["polarity"] + 1) / 2).values.reshape(-1, 1)
    X = np.hstack([vectors, pol])
    print(f"   Final X shape  : {X.shape}")
    return X

# ------------------------------------------------------------------
# 5. Clustering
# ------------------------------------------------------------------
def cluster(X, df):
    print("5. Finding best number of clusters (silhouette score)...")
    best_k, best_score = 5, -1
    for k in N_CLUSTERS_RANGE:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"   k={k}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"   → Selected best k = {best_k}")
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df["cl_num"] = km.fit_predict(X)
    return df, best_k

# ------------------------------------------------------------------
# 6. Unique tweets + Phrase Extraction
# ------------------------------------------------------------------
def unique_tweets(df):
    print("6. Creating unique tweet set...")
    u = (
        df.groupby(["tweet", "fully_cleaned_tweet", "sentiment", "cl_num"])
        .size()
        .reset_index(name="freq")
        .sort_values(["cl_num", "freq"], ascending=[True, False])
    )
    print(f"   Unique tweets  : {len(u)}")
    return u

def extract_phrases(df_unique):
    print("7. Extracting characteristic phrases (narratives)...")
    rows = []
    for cid in sorted(df_unique["cl_num"].unique()):
        texts = df_unique.loc[df_unique["cl_num"] == cid, "fully_cleaned_tweet"]
        all_tokens = []
        for t in texts:
            all_tokens.extend(word_tokenize(t))

        for bg, cnt in Counter(ngrams(all_tokens, 2)).most_common(12):
            if cnt >= 4:
                rows.append({
                    "abstraction": " ".join(bg),
                    "cl_num": int(cid),
                    "freq": cnt
                })
    narrative = pd.DataFrame(rows)
    print(f"   Phrases found  : {len(narrative)}")
    return narrative

# ------------------------------------------------------------------
# 7. Assign sentiment expression to narratives
# ------------------------------------------------------------------
def mode_sentiment(values):
    if not values:
        return 3
    c = Counter(values)
    most = c.most_common()
    if len(most) == 1:
        return most[0][0]
    if most[0][1] == most[1][1]:
        non_n = [m[0] for m in most[:2] if m[0] != 3]
        return non_n[0] if non_n else 3
    return most[0][0]

def assign_expression(narrative, df_unique):
    print("8. Assigning Positive / Negative / Neutral to each narrative...")
    codes = []
    for _, row in narrative.iterrows():
        phrase = row["abstraction"]
        cid = row["cl_num"]
        mask = (
            (df_unique["cl_num"] == cid) &
            (df_unique["fully_cleaned_tweet"].str.contains(re.escape(phrase), na=False))
        )
        sents = df_unique.loc[mask, "sentiment"].tolist()
        codes.append(mode_sentiment(sents))

    narrative = narrative.copy()
    narrative["expression_code"] = codes
    narrative["expression"] = narrative["expression_code"].map(
        {1: "Positive", 2: "Negative", 3: "Neutral"}
    )
    return narrative

# ------------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------------
def save_everything(narrative, df_unique):
    print("\n" + "=" * 60)
    print("FINAL NARRATIVES")
    print("=" * 60)
    display = (
        narrative[["cl_num", "abstraction", "expression", "freq"]]
        .sort_values(["cl_num", "freq"], ascending=[True, False])
    )
    print(display.to_string(index=False))

    # Bar chart
    plt.figure(figsize=(8, 5))
    order = ["Positive", "Negative", "Neutral"]
    sns.countplot(data=narrative, x="expression", order=order, palette="viridis")
    plt.title("Sentiment Distribution of Extracted Narratives")
    plt.ylabel("Number of Narratives")
    plt.tight_layout()
    plt.savefig("narrative_sentiment_count.png", dpi=150)
    print("\n✓ Saved : narrative_sentiment_count.png")

    # Excel report
    with pd.ExcelWriter("narrative.xlsx", engine="openpyxl") as writer:
        narrative.to_excel(writer, sheet_name="All_Narratives", index=False)
        for cid in sorted(df_unique["cl_num"].unique()):
            part = df_unique[df_unique["cl_num"] == cid][
                ["tweet", "freq", "fully_cleaned_tweet"]
            ].head(40)
            part.to_excel(writer, sheet_name=f"Cluster_{cid}", index=False)
    print("✓ Saved : narrative.xlsx")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    df = load_data()
    df = preprocess(df)
    df = add_sentiment(df)
    X = build_features(df)
    df, _ = cluster(X, df)
    df_unique = unique_tweets(df)
    narrative = extract_phrases(df_unique)
    narrative = assign_expression(narrative, df_unique)
    save_everything(narrative, df_unique)
    print("\n✅ Analysis completed successfully!")
    print("   Ready for GitHub / HR submission.")

if __name__ == "__main__":
    main()
