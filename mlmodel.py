import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset
emails = pd.read_csv("D:\Study\sem 6\ML\lab_7\spam.csv", encoding='latin-1')
emails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
emails.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)


# Preprocessing
# Preprocessing
# Preprocessing
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

emails['message'] = emails['message'].apply(remove_stop_words)

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

emails['message'] = [lemmatize_sentence(sentence) for sentence in emails['message']]

X_train, X_test, y_train, y_test = train_test_split(emails['message'], emails['label'].values, test_size=0.3, random_state=42)

# Convert the labels to numerical values using LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

#model for count vectorizer;
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

y_pred = classifier.predict(X_test_counts)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)


print("Accuracy of Naive bayes: ", accuracy)
print("Confusion Matrix for Naive bayes: ", confusion)

with open('D:\Study\sem 6\ML\lab_8\model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

with open('D:\Study\sem 6\ML\lab_8\ectoriser.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

