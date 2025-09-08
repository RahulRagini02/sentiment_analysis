from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from collections import Counter
from sentiment_analysis.preprocess_sentiment_analysis import preprocess_dataframe, wordcloud_to_base64
import seaborn as sns

app = FastAPI(title="Sentiment Analysis API")

API_KEY = "mysecret1234"  # Replace with your secret key

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

def ngrams_from_text(text: str, n: int):
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# ---------------- Helper: Compute Clause Sentiment Percentages ----------------
def compute_clause_sentiment(df: pd.DataFrame, category_col: str):
    """Returns a dictionary {clause: {Positive, Neutral, Negative}}"""
    if "Clause" not in df.columns or df["Clause"].dropna().empty:
        return None

    clause_data = {}
    for clause, group in df.groupby("Clause"):
        counts = group[category_col].value_counts(normalize=True) * 100
        clause_data[clause] = {
            "Positive": round(counts.get("Positive", 0), 2),
            "Neutral": round(counts.get("Neutral", 0), 2),
            "Negative": round(counts.get("Negative", 0), 2)
        }
    return clause_data

# ---------------- Helper: Generate Clause Chart Base64 ----------------
def clause_sentiment_chart(clause_data: dict):
    if not clause_data:
        return None
    chart_df = pd.DataFrame(clause_data).T[["Positive","Neutral","Negative"]]
    n_clauses = len(chart_df)
    index = range(n_clauses)
    bar_width = 0.25

    plt.figure(figsize=(10, max(4, n_clauses*0.5)))
    plt.barh([i - bar_width for i in index], chart_df["Positive"], height=bar_width, color="green", label="Positive")
    plt.barh(index, chart_df["Neutral"], height=bar_width, color="gray", label="Neutral")
    plt.barh([i + bar_width for i in index], chart_df["Negative"], height=bar_width, color="red", label="Negative")
    plt.yticks(index, chart_df.index)
    plt.xlabel("Percentage (%)")
    plt.ylabel("Clause")
    plt.title("Clause Sentiment Distribution")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------------- Main Analyze Endpoint ----------------
@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    # Preprocess
    df_processed = preprocess_dataframe(df)

    category_col = next((c for c in ["Label","label","Category","category"] if c in df_processed.columns), None)
    if not category_col:
        raise HTTPException(status_code=400, detail="No sentiment/category column found in CSV")

    filtered_df = df_processed.copy()

    # --- Overall Analysis ---
    all_text = " ".join(filtered_df["clean_comment"].astype(str).tolist())
    sentiment_counts = filtered_df[category_col].value_counts().to_dict()

    analysis = {}
    analysis["sentiment_counts"] = sentiment_counts
    analysis["top_words"] = Counter(all_text.split()).most_common(50)
    analysis["top_bigrams"] = Counter(ngrams_from_text(all_text, 2)).most_common(25)
    analysis["top_trigrams"] = Counter(ngrams_from_text(all_text, 3)).most_common(25)
    analysis["wordcloud_base64"] = wordcloud_to_base64(all_text)

    analysis["total_comments"] = len(filtered_df)
    analysis["unique_clause"] = int(filtered_df["Clause"].nunique()) if "Clause" in filtered_df.columns else 0
    analysis["avg_word_count"] = float(filtered_df["clean_comment"].str.split().str.len().mean())
    analysis["sample_processed"] = filtered_df[["clean_comment", category_col, "Clause"]].head(10).to_dict(orient="records")

    # --- Clause Sentiment Data & Chart ---
    clause_data = compute_clause_sentiment(filtered_df, category_col)
    analysis["clause_sentiment"] = clause_data
    analysis["clause_sentiment_chart_base64"] = clause_sentiment_chart(clause_data)

    return JSONResponse(content=analysis)

@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment Analysis API is running."}
