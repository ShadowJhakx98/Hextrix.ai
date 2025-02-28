"""
toy_text_gen.py

A small RNN-based text generation model on a tiny dataset.
In practice, you'd need a much larger dataset and a more advanced model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

class ToyTextGenerator:
    def __init__(self, sample_texts=None, epochs=2):
        """
        sample_texts: list of example text strings
        epochs: how many epochs to train on the tiny set
        """
        if sample_texts is None:
            sample_texts = [
                "hello world",
                "deep learning is fun",
                "we are building a small rnn for text generation",
            ]
        self.sample_texts = sample_texts
        self.epochs = epochs
        self.tokenizer = None
        self.model = None
        self._train()

    def _train(self):
        # Tokenize
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.sample_texts)
        sequences = self.tokenizer.texts_to_sequences(self.sample_texts)
        X = pad_sequences(sequences, padding='pre')
        y = np.array([0,1,1])  # dummy labels for a toy example

        # Build a small RNN
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=10, input_length=X.shape[1]),
            SimpleRNN(32),
            Dense(vocab_size, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(X, y, epochs=self.epochs)

    def generate_text(self, seed: str, num_words=5) -> str:
        """
        Predict num_words next tokens from the 'seed' text.
        """
        for _ in range(num_words):
            seq = self.tokenizer.texts_to_sequences([seed])
            seq = pad_sequences(seq, maxlen=5, padding='pre')  # 5 is arbitrary
            pred = self.model.predict(seq)
            pred_idx = np.argmax(pred, axis=1)[0]
            next_word = self.tokenizer.index_word.get(pred_idx, '')
            seed += ' ' + next_word
        return seed
