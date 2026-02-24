import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Initializing Spam SMS Classifier...")
    
    # 1. Create a small mock dataset
    data = {
        'text': [
            "Win a FREE iPhone 15 now! Click the link below.",
            "Hey, are we still meeting for lunch at 1 PM?",
            "URGENT: Your bank account has been restricted. Reply to verify.",
            "Can you send me the notes from yesterday's lecture?",
            "CONGRATULATIONS! You have been selected for a $1000 Walmart gift card.",
            "I'm running about 10 minutes late, start without me.",
            "Exclusive offer just for you! Get 80% off your next purchase.",
            "Did you figure out the solution to the last math problem?"
        ],
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
    }
    
    df = pd.DataFrame(data)
    print(f"Dataset loaded with {len(df)} samples.\n")

    # 2. Feature Extraction: Convert text to word count vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 4. Train the Naive Bayes Model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 5. Make Predictions and Evaluate
    predictions = model.predict(X_test)
    
    print("Evaluation Results\n" + "="*30)
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions))

    # 6. Test with a custom message
    test_message = ["Get a free vacation, just text back YES!"]
    test_vec = vectorizer.transform(test_message)
    test_pred = model.predict(test_vec)
    print(f"\nCustom Test Message: '{test_message[0]}'")
    print(f"Predicted Category: {test_pred[0].upper()}")

if __name__ == "__main__":
    main()
