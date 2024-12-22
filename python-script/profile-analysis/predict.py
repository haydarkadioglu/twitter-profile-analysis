import re
import nltk
from nltk.corpus import stopwords
import numpy as np


nltk.download("stopwords")
def procces_sentence(review:str)->str:
    sample_review = re.sub(r'http\S+', '', review)
    sample_review = re.sub("[^a-zA-Z]",' ',sample_review)
    sample_review = sample_review.lower()
    sample_review = sample_review.split()
    swords = set(stopwords.words("english"))                     
    sample_review = [w for w in sample_review if w not in swords]        
    sample_review = " ".join(sample_review)
    return sample_review



def predict_sentiment(sentence: str, model, vectorizer, catboost=False):
    processed_sentence = procces_sentence(sentence)
    vectorized_sentence = vectorizer.transform([processed_sentence]).toarray()
    prediction = model.predict(vectorized_sentence)
    probabilities = model.predict_proba(vectorized_sentence)
    
    sentiment_map = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}

    if catboost:
        predicted_class = int(prediction[0][0])
    else:
        predicted_class = prediction[0]

    predicted_sentiment = sentiment_map[predicted_class]

    max_probability = max(probabilities[0]) * 100

    return predicted_sentiment, max_probability
