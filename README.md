# Spam Message Detection System

## Project Overview

This project develops a machine learning system to classify SMS messages as either spam or ham (not spam).

## Dataset

The project uses a labelled SMS spam dataset containing 5,572 messages.

- Ham messages: 4,825
- Spam messages: 747

### Dataset Source

The dataset is based on the SMS Spam Collection dataset from the UCI Machine Learning Repository.

The dataset file used in this project contains 5,572 messages after obtaining the project copy used for this implementation.

## Methodology

1. Load the SMS dataset using Pandas.
2. Select the message and label columns.
3. Convert labels:
   - Ham = 0
   - Spam = 1
4. Preprocess the message text by:
   - Converting text to lowercase
   - Removing punctuation
   - Removing extra spaces
5. Split the dataset into training and testing sets.
6. Convert messages into numerical features using TF-IDF.
7. Train two machine learning models:
   - Multinomial Naive Bayes
   - Logistic Regression
8. Evaluate both models using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
9. Use Logistic Regression in the Streamlit application for message classification.

## Model Results

### Multinomial Naive Bayes

- Accuracy: 95.16%
- Precision: 100.00%
- Recall: 63.76%
- F1-score: 77.87%

### Logistic Regression

- Accuracy: 96.59%
- Precision: 99.12%
- Recall: 75.17%
- F1-score: 85.50%

Logistic Regression achieved the higher F1-score and was selected for the final application.

## Application

A Streamlit web interface allows users to enter an SMS message and receive a prediction:

- SPAM
- HAM (not spam)

## Project Files

- `spam.csv` - SMS dataset
- `check_data.py` - Dataset inspection
- `preprocess.py` - Text preprocessing
- `train_model.py` - Model training and evaluation
- `app.py` - Streamlit application
- `spam_model.pkl` - Saved Logistic Regression model
- `tfidf_vectorizer.pkl` - Saved TF-IDF vectorizer
- `requirements.txt` - Required Python packages
- `.gitignore` - Files excluded from Git
## How to Run

1. Create and activate a Python virtual environment.

2. Install the required packages:

```bash
pip install -r requirements.txt

3. Run the Streamlit application:
```bash
streamlit run app.py

4. Open the local URL shown in the terminal, usually:
```text
http://localhost:8501

