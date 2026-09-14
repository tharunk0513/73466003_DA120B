import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Message Detection System",
    page_icon="📩",
    layout="wide"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

data = pd.read_csv("spam.csv", encoding="latin-1")

data = data[["v1", "v2"]]

data.columns = ["label", "message"]

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# --------------------------------------------------
# Load Trained Model and TF-IDF Vectorizer
# --------------------------------------------------

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# --------------------------------------------------
# Message Preprocessing Function
# --------------------------------------------------

def preprocess_message(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------------------------------
# Dashboard Title
# --------------------------------------------------

st.title("📩 Spam Message Detection System")

st.write(
    "An NLP-based machine learning system for classifying SMS messages as "
    "Ham or Spam."
)

st.divider()

# --------------------------------------------------
# Dataset Statistics
# --------------------------------------------------

total_messages = len(data)
ham_messages = (data["label"] == 0).sum()
spam_messages = (data["label"] == 1).sum()
spam_percentage = (spam_messages / total_messages) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Messages",
        f"{total_messages:,}"
    )

with col2:
    st.metric(
        "Ham Messages",
        f"{ham_messages:,}"
    )

with col3:
    st.metric(
        "Spam Messages",
        f"{spam_messages:,}"
    )

with col4:
    st.metric(
        "Spam Percentage",
        f"{spam_percentage:.1f}%"
    )

st.divider()

# --------------------------------------------------
# Dataset Distribution
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

with col1:

    labels = ["Ham", "Spam"]
    values = [ham_messages, spam_messages]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_title("Message Class Distribution")
    ax.set_ylabel("Number of Messages")

    st.pyplot(fig)

with col2:

    st.write("### Dataset Summary")

    summary_data = pd.DataFrame({
        "Class": ["Ham", "Spam"],
        "Messages": [ham_messages, spam_messages]
    })

    st.dataframe(
        summary_data,
        width="stretch",
        hide_index=True
    )

    st.write(
        "The dataset contains more ham messages than spam messages, "
        "so the classes are imbalanced."
    )

st.divider()

# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.subheader("Model Performance Comparison")

model_names = [
    "Naive Bayes",
    "Logistic Regression"
]

accuracy_values = [
    95.16,
    96.59
]

precision_values = [
    100.00,
    99.12
]

recall_values = [
    63.76,
    75.17
]

f1_values = [
    77.87,
    85.50
]

performance_data = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracy_values,
    "Precision": precision_values,
    "Recall": recall_values,
    "F1-score": f1_values
})

st.dataframe(
    performance_data,
    width="stretch",
    hide_index=True
)

fig, ax = plt.subplots(figsize=(10, 5))

x = range(len(model_names))
width = 0.2

ax.bar(
    [i - 1.5 * width for i in x],
    accuracy_values,
    width,
    label="Accuracy"
)

ax.bar(
    [i - 0.5 * width for i in x],
    precision_values,
    width,
    label="Precision"
)

ax.bar(
    [i + 0.5 * width for i in x],
    recall_values,
    width,
    label="Recall"
)

ax.bar(
    [i + 1.5 * width for i in x],
    f1_values,
    width,
    label="F1-score"
)

ax.set_xticks(list(x))
ax.set_xticklabels(model_names)

ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison")

ax.set_ylim(0, 110)

ax.legend()

st.pyplot(fig)

st.info(
    "Logistic Regression was selected because it achieved the higher "
    "F1-score of 85.50%, compared with 77.87% for Naive Bayes."
)

st.divider()

# --------------------------------------------------
# Selected Model Results
# --------------------------------------------------

st.subheader("Selected Model: Logistic Regression")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "96.59%")

with col2:
    st.metric("Precision", "99.12%")

with col3:
    st.metric("Recall", "75.17%")

with col4:
    st.metric("F1-score", "85.50%")

st.divider()

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

st.subheader("Logistic Regression Confusion Matrix")

confusion_matrix_data = pd.DataFrame(
    [
        [965, 1],
        [37, 112]
    ],
    index=["Actual Ham", "Actual Spam"],
    columns=["Predicted Ham", "Predicted Spam"]
)

st.dataframe(
    confusion_matrix_data,
    width="stretch"
)

st.write(
    "The confusion matrix shows how the Logistic Regression model "
    "classified the test messages."
)

st.divider()

# --------------------------------------------------
# Spam Message Detector
# --------------------------------------------------

st.subheader("🔍 Try the Spam Message Detector")

st.write(
    "Enter an SMS message below and the trained Logistic Regression "
    "model will classify it."
)

message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize!"
)

if st.button("Check Message", type="primary"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        clean_message = preprocess_message(message)

        message_tfidf = vectorizer.transform([clean_message])

        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:

            st.error("🚨 This message is SPAM.")

        else:

            st.success("✅ This message is HAM (not spam).")

st.divider()

# --------------------------------------------------
# Project Information
# --------------------------------------------------

st.subheader("Project Information")

col1, col2 = st.columns(2)

with col1:

    st.write("**Dataset:** SMS Spam Collection")
    st.write("**Total Messages:** 5,572")
    st.write("**Machine Learning:** Classification")
    st.write("**Feature Extraction:** TF-IDF")

with col2:

    st.write("**Models Tested:**")
    st.write("- Multinomial Naive Bayes")
    st.write("- Logistic Regression")
    st.write("**Deployment:** Streamlit")

st.caption(
    "Spam Message Detection System | Individual Data Analytics Project"
)