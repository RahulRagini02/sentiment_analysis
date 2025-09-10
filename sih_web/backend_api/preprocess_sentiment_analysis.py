import pandas as pd
import re
import string
from wordcloud import WordCloud
import base64
import io

def preprocess_dataframe(df: pd.DataFrame):
    """
    Cleans and preprocesses the text column of a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with a new 'clean_comment' column.
    """
    df_copy = df.copy()
    comment_col = next((c for c in ["Comment", "comment", "Text", "text"] if c in df_copy.columns), None)
    if not comment_col:
        raise ValueError("DataFrame must contain a 'Comment' or 'Text' column.")
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text.strip()

    df_copy["clean_comment"] = df_copy[comment_col].apply(clean_text)
    return df_copy

def wordcloud_to_base64(text: str):
    """
    Generates a word cloud from text and encodes it as a base64 string.
    
    Args:
        text (str): The input text for the word cloud.
        
    Returns:
        str: The base64 encoded string of the word cloud image.
    """
    if not text.strip():
        return ""
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")