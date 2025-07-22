import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(str(text))
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_.isalpha()]
    return " ".join(tokens)

# Load training and test data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Pre-process
train_df['processed'] = train_df['sentence'].apply(preprocess_text)
test_df['processed'] = test_df['sentence'].apply(preprocess_text)

# Drop empty processed sentences
train_df = train_df[train_df['processed'].str.len() > 0]
test_df = test_df[test_df['processed'].str.len() > 0]

# TF-IDF and model
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000, min_df=2)
X_train = vectorizer.fit_transform(train_df['processed'])
y_train = train_df['label']

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
X_test = vectorizer.transform(test_df['processed'])
y_test = test_df['label']
y_pred = clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
