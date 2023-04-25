import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from autocorrect import Speller
import nltk
from flask import Flask, render_template, request, send_file
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')


app = Flask(__name__)
model = joblib.load("spam_classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")


@app.route('/')
def index():
    return render_template('index.html')


# Define preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
spell = Speller(lang='en')


def preprocess(text):
    # Convert to lower case
    text = text.lower()
    # Remove stop words
    words = word_tokenize(text)
    words = [word for word in words if not word in stop_words]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    # Stem words
    # words = [stemmer.stem(word) for word in words]
    # Correct spelling mistakes
    words = [spell(word) for word in words]
    return ' '.join(words)


def remove_stop_words(doc):
    words = word_tokenize(doc)
    filtered_words = [
        word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


lemmatizer = WordNetLemmatizer()


def lemmatize_sentence(sentence):
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)
    # Lemmatize each word in the sentence
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    # Join the lemmatized words back into a sentence
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get("message")
    # file = request.files['file']
    file = request.files['file']

    if message:
        # Preprocess the input message
        message = preprocess(message)
        message_vector = vectorizer.transform([message])

        # Make a prediction
        prediction = model.predict(message_vector)[0]

        # Return the prediction as JSON
        return render_template("index.html", prediction=prediction)

    else:
        df = pd.read_csv(file, encoding='latin-1')
        df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

        df['message'] = df['message'].apply(remove_stop_words)

        df['message'] = [lemmatize_sentence(sentence)
                         for sentence in df['message']]

        inp = vectorizer.transform(df['message'])
        prediction = model.predict(inp)
        data = {'message': df['message'],
                'lebel': prediction}

        # Create DataFrame
        cf = pd.DataFrame(data)
        # Save the output.
        cf.to_csv('pre.csv')
        return send_file('pre.csv')


if __name__ == '__main__':
    app.run(debug=True)
