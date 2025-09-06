import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

st.sidebar.title("Reddit Sentiment Analyzer")

API_URL = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000")

uploaded_file = st.sidebar.file_uploader("Upload Reddit CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded File")
    st.dataframe(df.head())

    # Let user choose Sentiment filter (Overall + unique labels from CSV if available)
    if "label" in df.columns:
        labels = df["label"].dropna().unique().tolist()
    else:
        labels = []
    labels.sort()
    labels.insert(0, "Overall")

    selected_label = st.sidebar.selectbox("Analyze sentiment", labels)

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Analyzing..."):
            files = {"file": ("uploaded.csv", uploaded_file.getbuffer(), "text/csv")}
            try:
                params = {"sentiment": None if selected_label == "Overall" else selected_label}
                resp = requests.post(API_URL.rstrip("/") + "/analyze", files=files, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        st.success(f"✅ Analysis Completed for **{selected_label}**")

        # ----- Top Stats -----
        st.title("📊 Top Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Total Comments")
            st.title(data.get("total_comments", 0))
        with col2:
            st.header("Unique Users")
            st.title(data.get("unique_users", 0))
        with col3:
            st.header("Avg. Word Count")
            st.title(round(data.get("avg_word_count", 0), 2))

        # ----- Sentiment Distribution -----
        if "sentiment_counts" in data:
            st.title("📈 Sentiment Distribution")
            sentiment_df = pd.DataFrame(
                list(data["sentiment_counts"].items()), columns=["Sentiment", "Count"]
            )

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(sentiment_df["Sentiment"], sentiment_df["Count"], color="teal")
                plt.xticks(rotation=30)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                ax.pie(
                    sentiment_df["Count"],
                    labels=sentiment_df["Sentiment"],
                    autopct="%0.1f%%",
                )
                ax.axis("equal")
                st.pyplot(fig)

        # ----- Top Words -----
        if "top_words" in data:
            st.title("🔠 Most Common Words")
            top_words = pd.DataFrame(data["top_words"], columns=["Word", "Frequency"])
            fig, ax = plt.subplots()
            ax.barh(top_words["Word"].head(15), top_words["Frequency"].head(15), color="orange")
            plt.xticks(rotation=30)
            st.pyplot(fig)

        # ----- Word Cloud -----
        if data.get("wordcloud_base64"):
            st.title("☁️ Word Cloud")
            img_b64 = data["wordcloud_base64"]
            img = base64.b64decode(img_b64)
            st.image(img, use_column_width=True)

        # ----- Processed Sample -----
        if "sample_processed" in data:
            st.title("📝 Sample Processed Data")
            st.dataframe(pd.DataFrame(data["sample_processed"]))

        # ----- Emoji / Reactions (if backend sends it) -----
        if "emoji_counts" in data:
            st.title("😂 Emoji Analysis")
            emoji_df = pd.DataFrame(data["emoji_counts"].items(), columns=["Emoji", "Count"])

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df.head(10))
            with col2:
                fig, ax = plt.subplots()
                ax.pie(
                    emoji_df["Count"].head(5),
                    labels=emoji_df["Emoji"].head(5),
                    autopct="%0.2f",
                )
                st.pyplot(fig)
