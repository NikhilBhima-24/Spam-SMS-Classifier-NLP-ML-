# Spam SMS Classifier (NLP)

A lightweight Natural Language Processing (NLP) application built with Python and Scikit-learn. This project demonstrates how to classify text messages as either `spam` (promotional/malicious) or `ham` (normal) using the Naive Bayes algorithm.

## Features
* **Machine Learning Pipeline:** Utilizes `CountVectorizer` for text feature extraction.
* **Algorithm:** Implements the `MultinomialNB` (Naive Bayes) classifier, which is highly effective for standard text classification.
* **Standalone Execution:** Includes a mock dataset directly within the script so no external CSV downloads are required to run and test the model.
* **Evaluation Metrics:** Outputs Model Accuracy, Precision, Recall, and F1-score via Scikit-learn's `classification_report`.

## Tech Stack
* **Python 3.x**
* **Scikit-learn** (Model training and evaluation)
* **Pandas** (Data manipulation)

## How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/NikhilBhima-24/spam-sms-classifier.git](https://github.com/NikhilBhima-24/spam-sms-classifier.git)
2. Install the required dependencies:

Bash
pip install pandas scikit-learn
Execute the script:

Bash
python spam_classifier.py
Output Example
The script will output the accuracy of the model on the test split and test a custom message at the end to demonstrate real-time prediction capabilities.
