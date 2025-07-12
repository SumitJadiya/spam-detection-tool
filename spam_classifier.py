# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 📌 Step 2: Load Dataset
file_path = './resources/assets/SMSSpamCollection.tsv'
df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])

print("Dataset Loaded Successfully!")
print(df.head())

# 📌 Step 2.1: Clean Data - Remove NaN values
print(f"\nDataset shape before cleaning: {df.shape}")
df = df.dropna()
print(f"Dataset shape after cleaning: {df.shape}")

# 📌 Step 3: Encode Labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label']) 

# 📌 Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_encoded'], test_size=0.2, random_state=42)

# 📌 Step 5: Vectorize Text with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 📌 Step 6: Train Classifier
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 📌 Step 7: Evaluate Model
y_pred = model.predict(X_test_vec)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# 📌 Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 📌 Step 9: Create Predict Function
def predict_spam(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return "Spam" if prediction[0] == 1 else "Ham"

# 📌 Step 10: Try Sample Predictions
samples = [
    "Congratulations! You’ve won a free iPhone. Click to claim now!",
    "Can we meet at 5 PM for coffee?",
    "URGENT! You have a pre-approved loan. Reply to get funds.",
    "I'll call you later tonight."
]

print("\nSample Predictions:\n")
for msg in samples:
    print(f"> {msg} \nPrediction: {predict_spam(msg)}\n")