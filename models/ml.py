import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import spacy

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_texts(df, nlp, text_col='sentence'):
    def preprocess(text):
        doc = nlp(str(text).lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)
    df['processed'] = df[text_col].apply(preprocess)
    return df

def train_and_evaluate(train_df, test_df, min_words=5):
    # Filter short sentences for improving performance
    train_df = train_df[train_df['sentence'].apply(lambda x: len(str(x).split()) >= min_words)]
    test_df = test_df[test_df['sentence'].apply(lambda x: len(str(x).split()) >= min_words)]

    print("Train class balance:\n", train_df['label'].value_counts())
    print("Test class balance:\n", test_df['label'].value_counts())

    # TF-IDF vectorization (with unigrams & bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train = vectorizer.fit_transform(train_df['processed'])
    X_test = vectorizer.transform(test_df['processed'])
    y_train = train_df['label']
    y_test = test_df['label']

    # train classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {acc:.3f}\n')
    print(classification_report(y_test, y_pred))

    # Show top features here
    feature_names = vectorizer.get_feature_names_out()
    if len(clf.classes_) == 2:
        top_idx = clf.coef_[0].argsort()[-10:][::-1]
        print(f"\nTop features for {clf.classes_[1]} (positive class):")
        for idx in top_idx:
            print(f"  {feature_names[idx]}")
        bottom_idx = clf.coef_[0].argsort()[:10]
        print(f"\nTop features for {clf.classes_[0]} (negative class):")
        for idx in bottom_idx:
            print(f"  {feature_names[idx]}")
    else:
        for i, class_label in enumerate(clf.classes_):
            top_idx = clf.coef_[i].argsort()[-10:][::-1]
            print(f"\nTop features for {class_label}:")
            for idx in top_idx:
                print(f"  {feature_names[idx]}")

    # Show prediction results for first 10 test samples
    print("\nSample predictions on test data:")
    sample_df = test_df.copy()
    sample_df['predicted'] = y_pred
    sample_df['correct'] = sample_df['label'] == sample_df['predicted']
    print(sample_df[['sentence', 'label', 'predicted', 'correct']].head(10).to_string(index=False))

def main():
    # Load spaCy
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    # Load train/test data
    train_df, test_df = load_data('data/processed/train.csv', 'data/processed/test.csv')

    # remove stop words & etc
    train_df = preprocess_texts(train_df, nlp)
    test_df = preprocess_texts(test_df, nlp)

    # train & evaluate 
    train_and_evaluate(train_df, test_df)

if __name__ == "__main__":
    main()
