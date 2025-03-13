import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download necessary NLTK resources (do this once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)  # For tokenization

def preprocess_text(text):
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and special characters
    tokens = nltk.word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w not in stop_words] # Remove stop words

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens] # Stemming

    return " ".join(stemmed_tokens)  # Return cleaned text as a string


# Load your dataset (replace with your actual file path)
try:
    df = pd.read_csv("your_dataset.csv", encoding="latin-1")  # Handle potential encoding issues
    # Ensure your CSV has columns named 'email' and 'label' (spam/ham)
except FileNotFoundError:
    print("Error: Dataset file not found. Please provide a valid CSV file.")
    exit()  # Exit if the file isn't found
except pd.errors.ParserError:
    print("Error: Could not parse CSV file. Please check file format.")
    exit()

# Preprocess the email text
df['processed_email'] = df['email'].apply(preprocess_text)

# Split data
X = df['processed_email']  # Features (preprocessed emails)
y = df['label']  # Labels (spam/ham)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random state for reproducibility


# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# Make predictions
predictions = classifier.predict(X_test_vectors)

# Evaluate performance
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# More detailed classification report
print(classification_report(y_test, predictions))


# Example of filtering a new email:
def filter_email(email_text):
    processed_email = preprocess_text(email_text)
    email_vector = vectorizer.transform([processed_email]) # Transform it to a vector
    prediction = classifier.predict(email_vector)[0] # Predict. [0] gets the label, not the numpy array
    return prediction

new_email = "This is a test email.  Claim your free prize today!"
prediction = filter_email(new_email)
print(f"The email is predicted to be: {prediction}")

new_email = "Meeting at 3pm tomorrow to discuss the project."
prediction = filter_email(new_email)
print(f"The email is predicted to be: {prediction}")