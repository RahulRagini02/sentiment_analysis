from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from collections import Counter
from preprocess import preprocess_dataframe, wordcloud_to_base64

app = FastAPI(title="Reddit Sentiment Analysis API")


def ngrams_from_text(text: str, n: int):
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


@app.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    sentiment: str = Form("overall")
):
    # --- Validate file ---
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    # --- Preprocess ---
    try:
        df_processed = preprocess_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in preprocessing: {e}")

    # --- Ensure category column ---
    category_col = None
    for col in ["category", "Category", "label", "Label"]:
        if col in df_processed.columns:
            category_col = col
            break
    if not category_col:
        raise HTTPException(status_code=400, detail="No sentiment/category column found in CSV")

    # --- Filtering ---
    sentiment = sentiment.lower()
    filtered_df = df_processed.copy()

    if sentiment == "overall":
        pass
    elif sentiment == "each":
        return JSONResponse(content={
            "mode": "each",
            "sample_processed": filtered_df[["clean_comment", category_col]].to_dict(orient="records")
        })
    elif sentiment in ["negative", "negatives", "objections"]:
        filtered_df = filtered_df[filtered_df[category_col].str.contains("negative|objection", case=False, na=False)]
    elif sentiment in ["positive", "positives", "strongly positive"]:
        filtered_df = filtered_df[filtered_df[category_col].str.contains("positive", case=False, na=False)]
    elif sentiment in ["neutral", "neutrals"]:
        filtered_df = filtered_df[filtered_df[category_col].str.contains("neutral", case=False, na=False)]
    else:
        # Match exact label
        filtered_df = filtered_df[filtered_df[category_col].astype(str).str.lower() == sentiment]

    # --- Always send sentiment_counts ---
    sentiment_counts = df_processed[category_col].value_counts().to_dict()
    filtered_counts = filtered_df[category_col].value_counts().to_dict()

    if filtered_df.empty:
        return JSONResponse(content={
            "error": f"No data found for sentiment '{sentiment}'",
            "sentiment_counts": sentiment_counts  # still return global counts
        })

    # --- Analysis results ---
    analysis = {}
    analysis["sentiment_counts"] = filtered_counts if filtered_counts else sentiment_counts

    # Word frequency
    all_text = " ".join(filtered_df["clean_comment"].astype(str).tolist())
    word_freq = Counter(all_text.split()).most_common(50)
    analysis["top_words"] = word_freq

    # N-grams
    analysis["top_bigrams"] = Counter(ngrams_from_text(all_text, 2)).most_common(25)
    analysis["top_trigrams"] = Counter(ngrams_from_text(all_text, 3)).most_common(25)

    # Wordcloud
    analysis["wordcloud_base64"] = wordcloud_to_base64(all_text)

    # Metadata
    analysis["total_comments"] = len(filtered_df)
    analysis["unique_users"] = int(filtered_df["author"].nunique()) if "author" in filtered_df.columns else 0
    analysis["avg_word_count"] = float(filtered_df["clean_comment"].str.split().str.len().mean())

    # Sample
    analysis["sample_processed"] = filtered_df[[ "clean_comment", category_col ]].head(10).to_dict(orient="records")

    return JSONResponse(content=analysis)


@app.get("/")
def root():
    return {"status": "ok", "message": "Reddit Sentiment Analysis API is running."}
