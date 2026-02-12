import streamlit as st
import joblib
import re
import string

# Load Model and Vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return text

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type a review (or any sentence), and the AI will tell you if it's Positive or Negative.")

# Text Input
user_input = st.text_area("Enter your review here:", "The movie was absolutely fantastic! I loved the acting.")

if st.button("Analyze Sentiment"):
    # 1. Clean the input
    cleaned_input = clean_text(user_input)
    
    # 2. Vectorize the input (Turn words to numbers)
    # Note: We use .transform(), NOT .fit_transform() because the vocab is already fixed
    vec_input = vectorizer.transform([cleaned_input]).toarray()
    
    # 3. Predict
    prediction = model.predict(vec_input)
    probability = model.predict_proba(vec_input)[0] # [Prob_Neg, Prob_Pos]

    # 4. Display
    st.divider()
    if prediction[0] == 1:
        st.success(f"😊 POSITIVE Review! (Confidence: {probability[1]:.2%})")
    else:
        st.error(f"😡 NEGATIVE Review. (Confidence: {probability[0]:.2%})")