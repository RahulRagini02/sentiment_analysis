import streamlit as st
import requests
import pandas as pd
import numpy as np
import io
from st_aggrid import AgGrid,GridOptionsBuilder
import matplotlib.pyplot as plt
import base64

st.set_page_config(layout="wide", page_title="SentimentAI")

# ---------------- API URLs ----------------
PREDICT_API = "http://127.0.0.1:8000/predict"
CLASSIFY_API = "http://127.0.0.1:8000/classify_file"
ANALYZE_API = "http://127.0.0.1:8500/analyze"

# ---------------- Session State ----------------
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "key_verified" not in st.session_state:
    st.session_state.key_verified = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "classified_df" not in st.session_state:
    st.session_state.classified_df = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ---------------- Sidebar ----------------
st.sidebar.title("🔑 SentimentAI – Secure Access")

# Always show API key input
api_key_input = st.sidebar.text_input(
    "Enter your API Key",
    type="password",
    placeholder="Enter your API key here..."
)

if st.sidebar.button("Submit API Key"):
    try:
        response = requests.post(
            CLASSIFY_API,
            files={},
            headers={"x-api-key": api_key_input},
            timeout=5
        )
        if response.status_code == 401:
            st.sidebar.error("❌ Invalid API key. Please try again.")
        else:
            st.session_state.api_key = api_key_input
            st.session_state.key_verified = True
            st.sidebar.success("✅ API key verified successfully!")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"⚠️ Unable to verify API key: {e}")

if not st.session_state.key_verified:
    st.warning("🔒 Please enter and submit your API key to unlock the app.")
    st.stop()

api_key = st.session_state.api_key
headers = {"x-api-key": api_key}

st.title("📊 SentimentAI Dashboard")

# ---------------- Text Prediction ----------------
st.subheader("🔹 Try with Text")
user_input = st.text_area("Type or paste text here...", height=120)

if st.button("Predict"):
    if user_input.strip():
        try:
            response = requests.post(
                PREDICT_API,
                json={"text": user_input},
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                st.session_state.prediction_result = response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"⚠️ API Error: {e}")
    else:
        st.warning("Please enter text.")

# Show prediction result if available
if st.session_state.prediction_result:
    data = st.session_state.prediction_result
    st.success("Prediction Complete ✅")

    sentiment = data.get("prediction", "N/A")
    confidence = round(data.get("Confidence", 0) * 100, 2)

    # Choose color based on sentiment
    if sentiment.lower() == "positive":
        color = "green"
    elif sentiment.lower() == "negative":
        color = "red"
    elif sentiment.lower() == "neutral":
        color = "orange"
    else:
        color = "black"

    col1, col2 = st.columns(2)
    
    # Sentiment with color
    col1.markdown(f"<h3 style='color:{color};'>{sentiment}</h3>", unsafe_allow_html=True)
    
    # Confidence as normal metric
    col2.metric("Confidence", f"{confidence}%")

# ---------------- File Upload ----------------
st.subheader("📂 Upload File for Classification")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
    if st.sidebar.button("Classify File"):
        response = requests.post(CLASSIFY_API, files=files, headers=headers)
        if response.status_code == 200:
            df = pd.read_csv(io.BytesIO(response.content))
            st.session_state.classified_df = df
            st.success("CSV classified successfully!")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

# Show classified DataFrame if available
if st.session_state.classified_df is not None:
    df = st.session_state.classified_df
    st.write("### Preview of Labeled CSV")
    AgGrid(df)

    st.download_button(
        label="Download Labeled CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="labeled_data.csv",
        mime="text/csv"
    )

    # ---------------- Clause filter ----------------
    if "Clause" in df.columns:
        clauses = sorted(df["Clause_id"].dropna().unique().tolist())
        clauses.insert(0, "Overall")
    else:
        clauses = ["Overall"]
    selected_clause = st.sidebar.selectbox("Choose Clause ID", clauses)

    # ---------------- Stakeholder filter ----------------
    if "Stakeholders" in df.columns:
        stakeholders = sorted(df["Stakeholders"].dropna().unique().tolist())
        stakeholders.insert(0, "Overall")
    else:
        stakeholders = ["Overall"]
    selected_stakeholder = st.sidebar.selectbox("Choose Stakeholder", stakeholders)

    # ---------------- Label filter ----------------
    if "Label" in df.columns:
        labels = sorted(df["Label"].dropna().unique().tolist())
        labels.insert(0, "Overall")
    else:
        labels = ["Overall"]
    selected_label = st.sidebar.selectbox("Choose Label", labels)

    # ---------------- Run Analysis ----------------
    if st.sidebar.button("Run Analysis"):
        with st.spinner("📊 Analyzing..."):
            # Apply filters step by step
            df_filtered = df.copy()

            if selected_clause != "Overall":
                df_filtered = df_filtered[df_filtered["Clause"] == selected_clause]
            if selected_stakeholder != "Overall":
                df_filtered = df_filtered[df_filtered["Stakeholders"] == selected_stakeholder]
            if selected_label != "Overall":
                df_filtered = df_filtered[df_filtered["Label"] == selected_label]

            if df_filtered.empty:
                st.warning("⚠️ No data found for the selected Clause/Stakeholder/Label.")
                st.stop()

            # Convert filtered DataFrame to CSV bytes
            csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
            files = {"file": ("filtered.csv", io.BytesIO(csv_bytes), "text/csv")}

            try:
                resp = requests.post(ANALYZE_API, files=files, headers={"x-api-key": api_key})
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        st.success("✅ Analysis Complete!")
        # st.write("### Filtered Dataset Preview")
        # st.dataframe(df_filtered)


        # ---------------- Top Statistics ----------------
        st.title("📊 Top Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Comments", data.get("total_comments", 0))
        col2.metric("Unique Clause", data.get("unique_clause", 0))
        col3.metric("Avg. Word Count", round(data.get("avg_word_count", 0), 2))

        # ---------------- Sentiment Distribution ----------------
        if "sentiment_counts" in data and data["sentiment_counts"]:
            st.title("📈 Sentiment Distribution")
            sentiment_df = pd.DataFrame(list(data["sentiment_counts"].items()), columns=["Sentiment", "Count"])
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.bar(sentiment_df["Sentiment"], sentiment_df["Count"], color="teal")
                plt.xticks(rotation=30)
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.pie(sentiment_df["Count"], labels=sentiment_df["Sentiment"], autopct="%0.1f%%")
                ax.axis("equal")
                st.pyplot(fig)
        else:
            st.warning("⚠️ No sentiment distribution data available.")

        # ---------------- Word Cloud ----------------
        if data.get("wordcloud_base64"):
            st.title("☁️ Word Cloud")
            img_b64 = data["wordcloud_base64"]
            img = base64.b64decode(img_b64)
            st.image(img, use_container_width=True)

        # ---------------- Most Common Words ----------------
        if "Most_words" in data:
            st.title("🔠 Most Common Words")
            top_words = pd.DataFrame(data["top_words"], columns=["Word", "Frequency"])
            fig, ax = plt.subplots()
            ax.barh(top_words["Word"].head(15), top_words["Frequency"].head(15), color="orange")
            ax.invert_yaxis()
            plt.xticks(rotation=30)
            st.pyplot(fig)

        # ---------------- N-Grams ----------------
        if "top_bigrams" in data or "top_trigrams" in data:
            st.title("📌 N-Gram Analysis")
            col1, col2 = st.columns(2)

            if "top_bigrams" in data:
                with col1:
                    st.subheader("Top Bigrams")
                    bigram_df = pd.DataFrame(data["top_bigrams"], columns=["Bigram", "Frequency"])
                    fig, ax = plt.subplots(figsize=(4, 3))  # slightly bigger than (3,2) for readability
                    ax.barh(bigram_df["Bigram"].head(10), bigram_df["Frequency"].head(10), color="blue")
                    ax.invert_yaxis()
                    st.pyplot(fig)

            if "top_trigrams" in data:
                with col2:
                    st.subheader("Top Trigrams")
                    trigram_df = pd.DataFrame(data["top_trigrams"], columns=["Trigram", "Frequency"])
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.barh(trigram_df["Trigram"].head(10), trigram_df["Frequency"].head(10), color="purple")
                    ax.invert_yaxis()
                    st.pyplot(fig)

        # ---------------- Horizontal Grouped Bar Chart ----------------
        if "clause_sentiment" in data and data["clause_sentiment"]:
            clause_data = data["clause_sentiment"]

            if isinstance(clause_data, dict) and len(clause_data) > 0:
                st.title("📊 Sentiment % by Clause")

                clause_df = pd.DataFrame(clause_data).T
                clause_df = clause_df[["Positive", "Neutral", "Negative"]]

                n_clauses = len(clause_df)
                index = np.arange(n_clauses)
                bar_width = 0.25

                fig, ax = plt.subplots(figsize=(7, max(4, n_clauses*0.5)))

                ax.barh(index - bar_width, clause_df["Positive"], height=bar_width, label="Positive", color="green")
                ax.barh(index, clause_df["Neutral"], height=bar_width, label="Neutral", color="grey")
                ax.barh(index + bar_width, clause_df["Negative"], height=bar_width, label="Negative", color="red")

                ax.set_yticks(index)
                ax.set_yticklabels(clause_df.index)
                ax.set_xlabel("Percentage (%)")
                ax.set_title("Sentiment Distribution per Clause")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Clause sentiment data is empty. Please check your CSV.")
        else:
            st.warning("⚠️ Clause sentiment data not available for the chart.")
