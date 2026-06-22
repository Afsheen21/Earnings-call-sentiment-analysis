import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_by_sentence(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return 0
    
    scores = [analyzer.polarity_scores(s)['compound'] for s in sentences]
    return sum(scores) / len(scores)

st.set_page_config(page_title="Earnings Call Sentiment Checker")
st.title("📊 Earnings Call Sentiment Checker")
st.write("Paste in earnings call text below to see its sentiment score.")

user_text = st.text_area("Paste earnings call transcript text here:", height=250)

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please paste some text first.")
    else:
        score = get_sentiment_by_sentence(user_text)
        
        st.subheader(f"Sentiment Score: {score:.3f}")
        
        if score > 0.3:
            st.success("This reads as fairly positive.")
        elif score > 0.15:
            st.info("This reads as mildly positive.")
        elif score > 0:
            st.write("This reads as roughly neutral.")
        else:
            st.error("This reads as negative.")
        
        st.caption("Note: based on my analysis, sentiment score showed no meaningful "
                   "correlation with actual stock price movement. This tool is for "
                   "demonstration only, not investment advice.")