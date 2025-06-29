import joblib

model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def predict_text(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]

print(predict_text('I absolutely love this!'))