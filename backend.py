from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from typing import Dict, Any
from collections import Counter

# Import your preprocessing function (make sure it's in preprocess.py)
from preprocess import preprocess_dataframe, wordcloud_to_base64


def plot_to_base64():
    # Create a simple plot
    plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
    plt.title("Example Plot")

    # Create an in-memory bytes buffer
    buf = io.BytesIO()

    # Save the plot to the buffer
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    # Move to the beginning of the buffer
    buf.seek(0)

    # Encode as base64
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return img_b64


# Example usage (you can comment this out in production)
if __name__ == "__main__":
    b64_string = plot_to_base64()
    print(b64_string[:100])  # Print only first 100 chars of the base64 string


app = FastAPI()


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    # Preprocess: detect language, translate to English, clean, lemmatize
    try:
        df_processed = preprocess_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in preprocessing: {e}")

    analysis: Dict[str, Any] = {}

    # 1) sentiment distribution if category exists
    if "category" in df_processed.columns:
        sentiment_counts = df_processed["category"].value_counts().to_dict()
        analysis["sentiment_counts"] = sentiment_counts

    # 2) word / ngram counts
    all_text = " ".join(df_processed["clean_comment"].astype(str).tolist())
    words = [w for w in all_text.split() if w.strip()]
    word_freq = Counter(words).most_common(50)
    analysis["top_words"] = word_freq

    # 3) bigrams/trigrams (simple)
    def ngrams_from_text(text: str, n: int):
        tokens = text.split()
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    bigrams = Counter(ngrams_from_text(all_text, 2)).most_common(25)
    trigrams = Counter(ngrams_from_text(all_text, 3)).most_common(25)
    analysis["top_bigrams"] = bigrams
    analysis["top_trigrams"] = trigrams

    # 4) wordcloud image base64
    img_b64 = wordcloud_to_base64(all_text)
    analysis["wordcloud_base64"] = img_b64

    # 5) simple metadata
    analysis["n_rows"] = len(df_processed)
    analysis["n_unique_comments"] = int(df_processed["clean_comment"].nunique())

    # Return processed sample (first 10 rows) for quick inspection
    analysis["sample_processed"] = (
        df_processed[
            ["clean_comment"] + [c for c in df_processed.columns if c != "clean_comment"]
        ]
        .head(10)
        .to_dict(orient="records")
    )

    return JSONResponse(content=analysis)


# Health check
@app.get("/")
def root():
    return {"status": "ok", "message": "Reddit Sentiment Analysis API is running."}
