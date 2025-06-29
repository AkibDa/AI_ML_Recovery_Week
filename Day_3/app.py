import streamlit as st
from sentiment_predictor import predict_text

st.title('🧠 Sentiment Analyzer')
text_input = st.text_area('Enter a sentence')

if st.button('Analyze'):
  result = predict_text(text_input)
  st.write(f"Sentiment: **{result.upper()}**")