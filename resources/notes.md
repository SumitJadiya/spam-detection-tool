⸻

🧾 Full Line-by-Line Explanation

⸻

📌 Step 1: Import Libraries

import pandas as pd
import numpy as np

	•	pandas is used for handling data in table (DataFrame) format.
	•	numpy is used for numerical operations (not directly used here but a common utility).

from sklearn.model_selection import train_test_split

	•	Used to split your dataset into training and testing sets.

from sklearn.feature_extraction.text import TfidfVectorizer

	•	Converts text into numbers using the TF-IDF method (explained below).

from sklearn.preprocessing import LabelEncoder

	•	Converts text labels like "spam" and "ham" into numbers (e.g., 1 and 0).

from sklearn.linear_model import LogisticRegression

	•	The ML model used here: Logistic Regression — good for binary classification (spam vs ham).

from sklearn.metrics import classification_report, confusion_matrix

	•	Tools to evaluate your model: accuracy, precision, recall, F1-score, and confusion matrix.

import seaborn as sns
import matplotlib.pyplot as plt

	•	Libraries to visualize the confusion matrix using a heatmap.

⸻

📌 Step 2: Load Dataset

url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

	•	Loads the dataset from a URL.
	•	sep='\t' means the data is tab-separated (TSV format).
	•	header=None tells pandas there’s no header row.
	•	names=['label', 'message'] assigns column names manually.

print("Dataset Loaded Successfully!")
print(df.head())

	•	Just confirming the dataset loaded correctly by printing the first 5 rows.

⸻

📌 Step 3: Encode Labels

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])  # spam = 1, ham = 0

	•	Converts "ham" and "spam" into 0 and 1 respectively so that the model can work with them.
	•	Adds a new column label_encoded to the dataframe.

⸻

📌 Step 4: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_encoded'], test_size=0.2, random_state=42)

	•	Splits the dataset into:
	•	Training data: 80% → used to train the model
	•	Testing data: 20% → used to test how well the model performs
	•	random_state=42 ensures repeatability (same split every time you run it).

⸻

📌 Step 5: Vectorize Text with TF-IDF

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

What is TF-IDF?
	•	TF = Term Frequency (how often a word appears in a document)
	•	IDF = Inverse Document Frequency (how rare the word is across all documents)
	•	It converts text messages into numerical vectors that represent how important each word is.
	•	stop_words='english' removes common useless words like “the”, “is”, “a”, etc.
	•	fit_transform is used on training data to learn the vocabulary and convert.
	•	transform is used on test data to apply the same transformation.

⸻

📌 Step 6: Train Classifier

model = LogisticRegression()
model.fit(X_train_vec, y_train)

	•	Creates and trains the Logistic Regression model on the vectorized training data.

⸻

📌 Step 7: Evaluate Model

y_pred = model.predict(X_test_vec)

	•	Uses the model to predict spam or ham for the test messages.

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

	•	Prints detailed model performance:
	•	Precision: out of all predicted spam, how many were correct?
	•	Recall: out of all actual spam, how many did we catch?
	•	F1 Score: balance between precision and recall

⸻

📌 Step 8: Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

	•	Creates a confusion matrix, which shows:
	•	True Positives (Spam correctly identified)
	•	True Negatives (Ham correctly identified)
	•	False Positives (Ham marked as Spam)
	•	False Negatives (Spam missed as Ham)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

	•	Plots the confusion matrix as a colored heatmap for easy visualization.

⸻

📌 Step 9: Create Predict Function

def predict_spam(text):
    vec = vectorizer.transform([text])     # Convert new text into same vector format
    prediction = model.predict(vec)        # Predict using the trained model
    return "Spam" if prediction[0] == 1 else "Ham"

	•	This is a helper function to classify any new message as “Spam” or “Ham”.
	•	It first vectorizes the input message and then predicts using the trained model.

⸻

📌 Step 10: Try Sample Predictions

samples = [
    "Congratulations! You’ve won a free iPhone. Click to claim now!",
    "Can we meet at 5 PM for coffee?",
    "URGENT! Your loan has been approved. Reply to get funds.",
    "I'll call you later tonight."
]

	•	A list of sample messages to test the model.

print("\nSample Predictions:\n")
for msg in samples:
    print(f"> {msg} \nPrediction: {predict_spam(msg)}\n")

	•	Loops through each message, runs it through the model, and prints whether it’s spam or not.
