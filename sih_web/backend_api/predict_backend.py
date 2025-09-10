from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import io
import pandas as pd
import uvicorn
import joblib
from fastapi.middleware.cors import CORSMiddleware
from preproces_predict_labels import preprocess_text

# ------------------ Load model & tokenizer ------------------
MODEL_PATH = "models/my_model"
LE_PATH = "models/label_encoder.pkl"

# Check if model and label encoder exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
    raise FileNotFoundError("Model files not found. Please place them in the 'models' directory.")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
label_encoder = joblib.load(LE_PATH)

app = FastAPI(title="DistilBERT Text Classification with Live Update")

# Add CORS Middleware here
origins = [
    "http://localhost",
    "http://localhost:3000", # The address where your React app is running
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

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    try:
        text_clean = preprocess_text(request.text)
        labels = ["Negative", "Neutral", "Positive"]
        inputs = tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        predicted_class = torch.argmax(scores).item()
        label = labels[predicted_class]
        predicted_score = round(float(scores[predicted_class]), 4)
        return {
            "prediction": label,
            "Confidence": predicted_score
        }
    except Exception as e:
        return {"error": str(e)}

def classify_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    scores = torch.nn.functional.softmax(output.logits, dim=1)[0]
    predicted_class = torch.argmax(scores).item()
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    predicted_score = round(float(scores[predicted_class]), 4)
    return predicted_label, predicted_score

@app.post("/classify_file")
async def classify_file(
    file: UploadFile = File(...),
    verified: bool = Depends(verify_api_key)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    filename = file.filename
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in [".csv", ".xls", ".xlsx", ".txt"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty.")

        if ext == ".csv":
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(io.BytesIO(contents))
        elif ext == ".txt":
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter="\t")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if "Comment" not in df.columns:
        raise HTTPException(status_code=400, detail="File must have a 'Comment' or 'Text' column.")

    if "ID" not in df.columns:
        df["ID"] = range(1, len(df) + 1)

    # The rest of your code for classification stays the same
    df["Comment"] = df["Comment"].apply(preprocess_text)
    labels = []
    scores = []
    for sentence in df["Comment"]:
        try:
            label, score = classify_text(sentence)
        except Exception:
            label, score = "Error", 0.0
        labels.append(label)
        scores.append(score)

    df["Label"] = labels
    df["Confidence"] = scores

    output_file = "labeled_data.csv"
    df.to_csv(output_file, index=False)

    return FileResponse(output_file, filename="labeled_data.csv", media_type='text/csv')


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/verify_key")
def verify_api_key_endpoint(verified: bool = Depends(verify_api_key)):
    return {"status": "ok", "message": "API Key is valid."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)