from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import numpy as np
import os
from preprocess_sentiment_analysis import preprocess_dataframe, wordcloud_to_base64

app = FastAPI(title="Sentiment Analysis API")

# Add CORS Middleware here
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "mysecret1234"

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

def ngrams_from_text(text: str, n: int):
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def compute_clause_sentiment(df: pd.DataFrame, category_col: str):
    if "Clause" not in df.columns or df["Clause"].dropna().empty:
        return {}
    clause_data = {}
    for clause, group in df.groupby("Clause"):
        counts = group[category_col].value_counts(normalize=True) * 100
        clause_data[clause] = {
            "Positive": round(counts.get("Positive", 0), 2),
            "Neutral": round(counts.get("Neutral", 0), 2),
            "Negative": round(counts.get("Negative", 0), 2)
        }
    return clause_data

def generate_chart_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_wordcount_distribution(df: pd.DataFrame, category_col: str):
    if "clean_comment" not in df.columns:
        return []
    df_wc = df.copy()
    df_wc["word_count"] = df_wc["clean_comment"].str.split().str.len()
    if df_wc[category_col].dtype != str:
        df_wc[category_col] = df_wc[category_col].astype(str)
    return df_wc[[category_col, "word_count"]].rename(columns={category_col: "category"}).to_dict(orient="records")

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    df_processed = preprocess_dataframe(df)

    category_col = next((c for c in ["Label", "label", "Category", "category"] if c in df_processed.columns), None)
    if not category_col:
        raise HTTPException(status_code=400, detail="No sentiment/category column found in CSV")

    all_text = " ".join(df_processed["clean_comment"].astype(str).tolist())

    analysis = {}
    analysis["sentiment_counts"] = df_processed[category_col].value_counts().to_dict()
    analysis["top_words"] = Counter(all_text.split()).most_common(50)
    analysis["top_bigrams"] = Counter(ngrams_from_text(all_text, 2)).most_common(25)
    analysis["top_trigrams"] = Counter(ngrams_from_text(all_text, 3)).most_common(25)
    analysis["wordcloud_base64"] = wordcloud_to_base64(all_text)

    analysis["total_comments"] = len(df_processed)
    analysis["unique_clause"] = int(df_processed["Clause"].nunique()) if "Clause" in df_processed.columns else 0
    analysis["avg_word_count"] = float(df_processed["clean_comment"].str.split().str.len().mean())
    analysis["sample_processed"] = df_processed[["clean_comment", category_col, "Clause"]].head(10).to_dict(orient="records")

    clause_data = compute_clause_sentiment(df_processed, category_col)
    analysis["clause_sentiment"] = clause_data
    
    # Generate charts as base64 images
    colors = {"Positive": "green", "Negative": "red", "Neutral": "orange"}

    # Bar chart
    fig, ax = plt.subplots(figsize=(4, 3))
    sentiment_df = pd.DataFrame(list(analysis["sentiment_counts"].items()), columns=["Sentiment", "Count"])
    ax.bar(sentiment_df["Sentiment"], sentiment_df["Count"], color=[colors.get(s, "blue") for s in sentiment_df["Sentiment"]])
    ax.set_title("Sentiment Counts")
    analysis["bar_chart_base64"] = generate_chart_base64(fig)

    # Pie chart
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(sentiment_df["Count"], labels=sentiment_df["Sentiment"], autopct="%0.1f%%", colors=[colors.get(s, "blue") for s in sentiment_df["Sentiment"]])
    ax.set_title("Sentiment Proportions")
    analysis["pie_chart_base64"] = generate_chart_base64(fig)

    # Top words
    fig, ax = plt.subplots(figsize=(5, 3))
    top_words_df = pd.DataFrame(analysis["top_words"], columns=["Word", "Frequency"])
    ax.barh(top_words_df["Word"].head(15), top_words_df["Frequency"].head(15), color="orange")
    ax.invert_yaxis()
    ax.set_title("Most Common Words")
    analysis["top_words_base64"] = generate_chart_base64(fig)

    # Bigrams
    fig, ax = plt.subplots(figsize=(4, 3))
    bigram_df = pd.DataFrame(analysis["top_bigrams"], columns=["Bigram", "Frequency"])
    ax.barh(bigram_df["Bigram"].head(10), bigram_df["Frequency"].head(10), color="blue")
    ax.invert_yaxis()
    ax.set_title("Top Bigrams")
    analysis["top_bigrams_base64"] = generate_chart_base64(fig)

    # Trigrams
    fig, ax = plt.subplots(figsize=(4, 3))
    trigram_df = pd.DataFrame(analysis["top_trigrams"], columns=["Trigram", "Frequency"])
    ax.barh(trigram_df["Trigram"].head(10), trigram_df["Frequency"].head(10), color="purple")
    ax.invert_yaxis()
    ax.set_title("Top Trigrams")
    analysis["top_trigrams_base64"] = generate_chart_base64(fig)

    # Word count distribution by sentiment
    word_count_data = generate_wordcount_distribution(df_processed, category_col)
    if word_count_data:
        wc_df = pd.DataFrame(word_count_data)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.kdeplot(data=wc_df, x='word_count', hue='category', ax=ax, fill=True)
        ax.set_title('Word Count Distribution by Sentiment Category')
        ax.set_xlabel('Word Count')
        analysis["word_count_kde_base64"] = generate_chart_base64(fig)
        
    # Top words by category
    if "clean_comment" in df_processed.columns and "Label" in df_processed.columns:
        comments_df = df_processed[["clean_comment", "Label"]].rename(columns={"Label": "category"})
        all_counts = {}
        for cat in comments_df["category"].unique():
            subset = comments_df[comments_df["category"] == cat]
            words = " ".join(subset["clean_comment"].astype(str)).split()
            word_freq = Counter(words)
            all_counts[cat] = dict(word_freq.most_common(20))
        top_words_df = pd.DataFrame(all_counts).fillna(0).astype(int)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        top_words_df.plot(kind="bar", stacked=True, ax=ax, color=[colors.get(c) for c in top_words_df.columns])
        ax.set_title("Top Words by Sentiment Category")
        analysis["top_words_by_sentiment_base64"] = generate_chart_base64(fig)

    return JSONResponse(content=analysis)

@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment Analysis API is running."}