# 🛡️ AI Message Guard

SMS spam/ham classifier built with TF-IDF + Multinomial Naive Bayes, deployed as a Streamlit web app.

🔗 **Live demo:** https://ai-message-guard-5am8t2d3vwpx8bfmiucr5w.streamlit.app

## Features
- Paste any message and get a SPAM/HAM prediction with confidence scores
- Model performance tab with accuracy, precision, recall, F1 and confusion matrix

## Results (test set, 20% split)
| Metric | Score |
|---|---|
| Accuracy | 97.57%|
| Spam Precision | 1.0 |
| Spam Recall | 80.4% |
| Spam F1 | 89.1% |

## Tech Stack
Python, scikit-learn, pandas, joblib, Streamlit

## Dataset
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

## Run locally
    pip install -r requirements.txt
    python train.py
    python -m streamlit run app.py

## Limitations
Educational project. It is an SMS spam classifier, not a phishing detector. A HAM prediction does not guarantee a message is safe.

## Author
Muhammad Shahnawaz
