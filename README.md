# 🤖 CheerBot

CheerBot is a machine learning-based chatbot that uses TF-IDF and SVM classification to answer FAQ-style queries.

## 💡 Features

- Cleans and lemmatizes input text
- Encodes answers using LabelEncoder
- Uses LinearSVC for training
- Predicts most relevant answers from a CSV FAQ list

## 📁 Files

- `test.py` – Main script for training and testing
- `chatbot.csv` – Dataset with Questions and Answers
- `requirements.txt` – List of Python dependencies

## 📌 Example Usage

```python
search_test = ["how to help my child who’s stressed about placements?"]
