import pandas as pd
from sklearn.metrics import classification_report

def get_keywords():
    sclc_keywords = [
        'small cell', 'etoposide', 'limited stage', 'extensive', 'sclc']
    nsclc_keywords = [
        'non small cell', 'egfr', 'pembrolizumab', 'osimertinib', 'nsclc', 'pd-l1', 'squamous', 'resectable']
    return sclc_keywords, nsclc_keywords

def predict_sentence(sentence, sclc_keywords, nsclc_keywords, default_class="NSCLC"):
    s = str(sentence).lower()
    sclc_hits = sum([kw in s for kw in sclc_keywords])
    nsclc_hits = sum([kw in s for kw in nsclc_keywords])
    if sclc_hits > nsclc_hits:
        return 'SCLC'
    elif nsclc_hits > sclc_hits:
        return 'NSCLC'
    else:
        return default_class

def predict(sentences, sclc_keywords, nsclc_keywords, default_class="NSCLC"):
    return [
        predict_sentence(s, sclc_keywords, nsclc_keywords, default_class)
        for s in sentences
    ]

def evaluate(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    sclc_keywords, nsclc_keywords = get_keywords()
    y_true = test_df['label']
    y_pred = predict(test_df['sentence'], sclc_keywords, nsclc_keywords)
    acc = (y_true == y_pred).mean()
    print(f"Naive rule-based accuracy on test set: {acc:.3f}")
    print(classification_report(y_true, y_pred))
    return acc

def main():
    evaluate('data/processed/test.csv')

if __name__ == "__main__":
    main()
