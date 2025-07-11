import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_text(train_df, test_df, max_words=10000, max_len=50):
    label_map = {'NSCLC': 0, 'SCLC': 1}
    y_train = train_df['label'].map(label_map).values
    y_test = test_df['label'].map(label_map).values
    X_train = train_df['sentence'].astype(str).tolist()
    X_test = test_df['sentence'].astype(str).tolist()

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')
    return X_train_seq, X_test_seq, y_train, y_test, X_test, tokenizer

def build_nn_model(max_words=10000, max_len=50):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_and_evaluate(model, X_train_seq, y_train, X_test_seq, y_test, X_test):
    callbacks = [EarlyStopping(patience=2, restore_best_weights=True)]
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )

    y_pred_prob = model.predict(X_test_seq)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print(f'\nTest Accuracy: {accuracy_score(y_test, y_pred):.3f}\n')
    print(classification_report(y_test, y_pred, target_names=['NSCLC','SCLC']))

    print("\nshow sample predictions:")
    for text, true, pred in zip(X_test[:10], y_test[:10], y_pred[:10]):
        print(f"Text: {text}\nTrue: {['NSCLC','SCLC'][true]}, Predicted: {['NSCLC','SCLC'][pred]}\n")
    return history

def main():
    train_path = 'data/processed/train.csv'
    test_path = 'data/processed/test.csv'
    max_words = 10000
    max_len = 50

    train_df, test_df = load_data(train_path, test_path)
    X_train_seq, X_test_seq, y_train, y_test, X_test, tokenizer = preprocess_text(train_df, test_df, max_words, max_len)
    model = build_nn_model(max_words, max_len)
    train_and_evaluate(model, X_train_seq, y_train, X_test_seq, y_test, X_test)

if __name__ == '__main__':
    main()
