import pickle
import nltk

# Load the saved vectorizer and model from the same folder
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    ps = nltk.PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)

# Function to predict if a message is spam or not
def predict_spam_or_not(text):
    transformed_text = transform_text(text)
    vectorized_text = cv.transform([transformed_text])
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return "SPAM."
    else:
        return "NOT SPAM."

if __name__ == "__main__":
    # Ask the user to input a message
    message = input("Enter a message: ")
    # Predict and print the result
    result = predict_spam_or_not(message)
    print(result)
