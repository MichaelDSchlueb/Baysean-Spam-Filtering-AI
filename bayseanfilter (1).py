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
nltk.download('punkt_tab')

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
    df = pd.read_csv("spam.csv", encoding="latin-1")  # Handle potential encoding issues
    # Ensure your CSV has columns named 'email' and 'label' (spam/ham)
except FileNotFoundError:
    print("Error: Dataset file not found. Please provide a valid CSV file.")
    exit()  # Exit if the file isn't found
except pd.errors.ParserError:
    print("Error: Could not parse CSV file. Please check file format.")
    exit()

# Preprocess the email text
df['processed_email'] = df['message'].apply(preprocess_text)

# Split data
X = df['processed_email']  # Features (preprocessed emails)
y = df['label']  # Labels (spam/ham)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Added random state for reproducibility


# Create TF-IDF vectors
vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
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

new_email = "watch anime now"
prediction = filter_email(new_email)
print(f" {new_email} is predicted to be: {prediction}")

new_email = "takeout and anime at my house"
prediction = filter_email(new_email)
print(f"{new_email} is predicted to be: {prediction}")

new_email = "sell me your anime collection"
prediction = filter_email(new_email)
print(f"{new_email} is predicted to be: {prediction}")

print("As you can see, the accuracy is 89 and 1/2, meaning that this machine learning function is pretty good at determining Ham from Spam")
print("Precision is the measure of how accurate a machine learning system is in determining what is spam vs what is ham (Memory vs. Recall vs. Precision).")
print("Recall is the ability of a machine learning system to identify the amount of predictions it got right based upon the actual results (Memory vs. Recall vs. Precision)")
print("As you can see, recall on spam is signficiantly lower than that of ham, meaning that while it is highly accurate in predicting spam, it cannot easily guess correctly what turned out to be spam.")
print("Overall, the Machine learning model is pretty good, except that it needs more training on recalling whether or not messages are spam.")
print("From the basic accuracy, you see that the precision of the machine learning model when it comes to ham is right on target with 0.89 while it appears be higher by .007 when it comes to spam")
print("The measure of recall from accuracy when looking at spam is a difference by 0.11 because recall is higher than the accuracy when it comes to Ham while there is a huge drop by 0.66.")
print("In terms of distance from each other, the precision for spam is higher than that of spam while ham has higher recall than spam. This shows that there is a negative correlation among each because not all abilities can be the same, as shown when looking at the .07 difference between ham and spam's respective precisions")
print("The distance between the two recalls is a difference by .77, showing that recall between spam and ham are inverses of each other.")
print("")
print("Work Cited:")
print("\"Accuracy vs. Recall vs. Precision: What's the Difference?\" Evidently AI. Accessed February 24 2025.")
print("https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall#:~:text=Recall%20is%20a%20metric%20that,the%20number%20of%20positive%20instances.")