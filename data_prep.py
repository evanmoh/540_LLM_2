# Evan Moh - Data Prep for Duke University 540 - LLM

import PyPDF2
import spacy
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def clean_for_excel(s):
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', s)

def strip_symbols(s):
    return re.sub(r'[•◊\-–—•●▪‣∙·]', '', s).strip()

def is_clinical_sentence(s):
    s = str(s).replace('\n', ' ').replace('\r', ' ').strip()
    if len(s.split()) < 6:
        return False
    # remove citation patterns happening.
    if re.match(r'^\d+(\.|,)?\s*[A-Za-z\s\-]+(et al\.|study|trial|cohort|Reference)?', s):
        return False
    citation_keywords = ['et al.', 'cohort study', 'trial', 'reference', 'clinicaltrials.gov']
    if any(kw in s.lower() for kw in citation_keywords):
        if len(s) < 60:
            return False
    bad_patterns = [
        "NCCN", "Version", "Copyright", "Guidelines Index", "All Rights Reserved",
        "Printed by", "Categories of Evidence", "Table of Contents",
        "Comprehensive Cancer Center", "Panel Members", "Discussion", "Updates",
        "Summary", "Index", "www.nccn.org", "All Rights Reserved"
    ]
    if any(pat.lower() in s.lower() for pat in bad_patterns):
        return False
    if re.fullmatch(r'[•.]*', s):
        return False
    return True

def extract_sentences(pdf_path, label):
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 5_000_000 
    with open(pdf_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    doc = nlp(text)
    sentences = [
        strip_symbols(clean_for_excel(sent.text.strip()))
        for sent in doc.sents if sent.text.strip()
    ]
    sentences_final = [s for s in sentences if is_clinical_sentence(s)]
    df = pd.DataFrame({'sentence': sentences_final})
    df['label'] = label
    return df

# Extract sentences from each PDF
df_nsclc = extract_sentences('data/raw/nsclc_nccn.pdf', 'NSCLC')
df_sclc  = extract_sentences('data/raw/sclc_nccn.pdf',  'SCLC')

# Combine and dedup
df_all = pd.concat([df_nsclc, df_sclc], ignore_index=True)
df_all['sentence_lower'] = df_all['sentence'].str.lower().str.strip()
df_all = df_all.drop_duplicates(subset=['sentence_lower', 'label'])
df_all = df_all.drop(columns=['sentence_lower'])
df_all = df_all.reset_index(drop=True)
df_all['idx'] = df_all.index  
df_all = df_all[['idx', 'sentence', 'label']]

# Add processed_sentence column for all.csv 

nlp = spacy.load('en_core_web_sm')


def spacy_preprocess(text):
    doc = nlp(str(text))
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.isalpha()]
    return " ".join(tokens)

tqdm.pandas()
df_all['processed_sentence'] = df_all['sentence'].progress_apply(spacy_preprocess)

# Save combined file
df_all.to_csv('all.csv', index=False)
print("all.csv created")

# split test and training set for ml models and nn models.
train_df, test_df = train_test_split(
    df_all[['idx', 'sentence', 'label']], 
    test_size=0.3,
    random_state=42,
    shuffle=True,
    stratify=df_all['label']
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("finished")
