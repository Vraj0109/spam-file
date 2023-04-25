from flask import Flask, render_template, request,send_file
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
# Load the model


model = pickle.load(open('D:\Study\sem 6\ML\lab_8\model.pkl', 'rb'))
vectorizer = pickle.load(open('D:\Study\sem 6\ML\lab_8\ectoriser.pkl', 'rb'))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_csv(file,encoding='latin-1')

    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

    # Preprocess the input
    # input_data = vectorizer.transform([text])
    # print(input_data.shape)

    # Make prediction using th.e loaded model
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    def remove_stop_words(doc):
        words = word_tokenize(doc)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    df['message'] = df['message'].apply(remove_stop_words)

    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define a function to lemmatize a sentence
    def lemmatize_sentence(sentence):
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        # Lemmatize each word in the sentence
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        # Join the lemmatized words back into a sentence
        lemmatized_sentence = ' '.join(lemmatized_words)
        return lemmatized_sentence

    df['message'] = [lemmatize_sentence(sentence) for sentence in df['message']]

    inp = vectorizer.transform(df['message'])
    prediction = model.predict(inp)
    data = {'message': df['message'],
		'lebel':prediction }

    # Create DataFrame
    cf = pd.DataFrame(data)

    # Save the output.
    cf.to_csv('D:\Study\sem 6\ML\lab_8\pre.csv')

    # Render the result template with the prediction
    return send_file('pre.csv')
    return render_template('index.html', prediction=prediction)
    # return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
