# 📧 Spam Detection Tool (ML Project)

This is a beginner-friendly Machine Learning project that detects whether a given SMS message is **spam** or **not spam (ham)** using **Logistic Regression** and **TF-IDF** vectorization.

## 🔍 Features
- Binary classification (spam vs ham)
- Real-world SMS Spam dataset
- Clean and interpretable model
- Simple command-line text prediction

## 📁 Project Structure

```
spam-detector/
├── resources/
│   └── asset/
│       └── SMSSpamCollection.tsv ✅
├── spam_classifier.py
├── requirements.txt
├── README.md
```

## Running this project

Activate virtual env 

```
python -m venv venv
source venv/bin/activate 
```

Install dependencies

```
pip install -r requirements.txt
```

Run
```
python spam_classifier.py
```