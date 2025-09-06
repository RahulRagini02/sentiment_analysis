import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

st.sidebar.title("📢 Reddit Sentiment Analyzer")

# FastAPI URL
API_URL="http://localhost:8000"

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Reddit CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preview uploaded data
    st.write("### Preview of Uploaded File")
    st.dataframe(df)

    # Choose sentiment filter
    if "Label" in df.columns:
        labels = df["Label"].dropna().unique().tolist()
    else:
        labels = []
    labels = sorted([str(x) for x in labels])
    labels.insert(0, "Overall")

    selected_label = st.sidebar.selectbox("Analyze sentiment", labels)

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Analyzing..."):

            # filter dataset based on label
            if selected_label != "Overall":
                df_filtered = df[df["Label"] == selected_label]
            else:
                df_filtered = df

            # Save filtered data to CSV (in memory)
            csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")

            files = {"file": ("filtered.csv", csv_bytes, "text/csv")}

            try:
                # we don’t need to send sentiment param anymore,
                # because filtering is already done here
                resp = requests.post(API_URL.rstrip("/") + "/analyze", files=files)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()
        st.success("Analysis Complete!")    

        # ---------------- Top Statistics ----------------
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

        # ---------------- Sentiment Distribution ----------------
        if "sentiment_counts" in data and data["sentiment_counts"]:
            st.title("📈 Sentiment Distribution")
            sentiment_df = pd.DataFrame(list(data["sentiment_counts"].items()), columns=["Sentiment", "Count"])

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(sentiment_df["Sentiment"], sentiment_df["Count"], color="teal")
                plt.xticks(rotation=30)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                ax.pie(sentiment_df["Count"], labels=sentiment_df["Sentiment"], autopct="%0.1f%%")
                ax.axis("equal")
                st.pyplot(fig)
        else:
            st.warning("⚠️ No sentiment distribution data available.")


       # ----- Word Cloud -----
        if data.get("wordcloud_base64"):
            st.title("☁️ Word Cloud")
            img_b64 = data["wordcloud_base64"]
            img = base64.b64decode(img_b64)
            st.image(img, use_container_width=True)  # fixed

        # ---------------- Most Common Words ----------------
        if "top_words" in data:
            st.title("🔠 Most Common Words")
            top_words = pd.DataFrame(data["top_words"], columns=["Word", "Frequency"])
            fig, ax = plt.subplots()
            ax.barh(top_words["Word"].head(15), top_words["Frequency"].head(15), color="orange")
            ax.invert_yaxis()
            plt.xticks(rotation=30)
            st.pyplot(fig)

        # ---------------- N-Grams ----------------
        if "top_bigrams" in data:
            st.title("📌 Top Bigrams")
            bigram_df = pd.DataFrame(data["top_bigrams"], columns=["Bigram", "Frequency"])
            fig, ax = plt.subplots()
            ax.barh(bigram_df["Bigram"].head(10), bigram_df["Frequency"].head(10), color="blue")
            ax.invert_yaxis()
            st.pyplot(fig)

        if "top_trigrams" in data:
            st.title("📌 Top Trigrams")
            trigram_df = pd.DataFrame(data["top_trigrams"], columns=["Trigram", "Frequency"])
            fig, ax = plt.subplots()
            ax.barh(trigram_df["Trigram"].head(10), trigram_df["Frequency"].head(10), color="purple")
            ax.invert_yaxis()
            st.pyplot(fig)

        # ---------------- Emoji Analysis ----------------
        if "emoji_counts" in data:
            st.title("😂 Emoji Analysis")
            emoji_df = pd.DataFrame(data["emoji_counts"].items(), columns=["Emoji", "Count"])

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df.head(10))
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df["Count"].head(5), labels=emoji_df["Emoji"].head(5), autopct="%0.2f")
                st.pyplot(fig)

        # # ---------------- Sample Processed Data ----------------
        # if "sample_processed" in data:
        #     st.title("📝 Sample Processed Data")
        #     st.dataframe(pd.DataFrame(data["sample_processed"]))
