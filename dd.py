tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('fake_news_model.pkl')

# Make sure you have the stopwords downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags (like <div>, <br>, etc.)
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # 4. Remove newline, tab characters
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # 5. Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # 6. Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    # 7. Join back to a clean sentence
    return ' '.join(tokens)

headline = "A woman whose photos were stolen to create the fake persona of 'Dr. Aisha' - a catfish Twitter account posing as a frontline doctor who died of COVID-19, is a medical student in South Africa"
cleaned = preprocess(headline)  # Use your cleaning function
vectorized = tfidf_vectorizer.transform([cleaned])
prediction = model.predict(vectorized)

print("Prediction:", "Real" if prediction[0]==1 else "Fake")