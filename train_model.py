import pandas as pd

data = pd.read_csv("spam.csv", encoding="latin-1")

data = data[["v1", "v2"]]
data.columns = ["label", "message"]

data["message"] = data["message"].str.lower()
data["message"] = data["message"].str.replace(r"[^\w\s]", "", regex=True)
data["message"] = data["message"].str.replace(r"\s+", " ", regex=True).str.strip()

data["label"] = data["label"].map({"ham": 0, "spam": 1})

print("Dataset loaded successfully!")
print("Number of rows:", len(data))
print(data.head())
X = data["message"]
y = data["label"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

print("Naive Bayes model trained successfully!")
y_pred_nb = nb_model.predict(X_test_tfidf)

print("Predictions generated successfully!")
print("Number of predictions:", len(y_pred_nb))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

print("Naive Bayes Accuracy:", accuracy_nb)
print("Naive Bayes Precision:", precision_nb)
print("Naive Bayes Recall:", recall_nb)
print("Naive Bayes F1-score:", f1_nb)
from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_test, y_pred_nb)

print("Naive Bayes Confusion Matrix:")
print(cm_nb)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_tfidf, y_train)

print("Logistic Regression model trained successfully!")
y_pred_lr = lr_model.predict(X_test_tfidf)

print("Logistic Regression predictions generated successfully!")
print("Number of predictions:", len(y_pred_lr))
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Precision:", precision_lr)
print("Logistic Regression Recall:", recall_lr)
print("Logistic Regression F1-score:", f1_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

print("Logistic Regression Confusion Matrix:")
print(cm_lr)
print("\nModel Comparison:")
print("Naive Bayes F1-score:", f1_nb)
print("Logistic Regression F1-score:", f1_lr)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Ham", "Spam"]
).plot()

plt.title("Logistic Regression Confusion Matrix")
plt.savefig("logistic_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
models = ["Naive Bayes", "Logistic Regression"]

accuracy_values = [accuracy_nb, accuracy_lr]
precision_values = [precision_nb, precision_lr]
recall_values = [recall_nb, recall_lr]
f1_values = [f1_nb, f1_lr]

x = range(len(models))
width = 0.2

plt.figure(figsize=(10, 6))

plt.bar([i - 1.5 * width for i in x], accuracy_values, width, label="Accuracy")
plt.bar([i - 0.5 * width for i in x], precision_values, width, label="Precision")
plt.bar([i + 0.5 * width for i in x], recall_values, width, label="Recall")
plt.bar([i + 1.5 * width for i in x], f1_values, width, label="F1-score")

plt.xticks(list(x), models)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1.1)
plt.legend()

plt.show()
import pickle

with open("spam_model.pkl", "wb") as file:
    pickle.dump(lr_model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully!")