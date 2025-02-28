"""
toy_tts.py

A naive "text-to-speech" neural net that outputs 1024 float samples,
not real audio. It's just a placeholder for a real TTS engine.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class ToyTTS:
    def __init__(self):
        self.model = self._build_model()
        # We'll do a tiny training with random data as a placeholder
        dummy_X = np.random.rand(5,10)
        dummy_y = np.random.rand(5,1024)
        self.model.fit(dummy_X, dummy_y, epochs=2, verbose=0)

    def _build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(1024, activation='linear')  # 1024 samples
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def speak(self, text: str):
        """
        Convert text -> small numeric vector -> neural net -> wave array
        """
        arr = np.zeros((1,10))
        # Convert up to first 10 chars of text
        for i,c in enumerate(text[:10]):
            arr[0,i] = ord(c) % 256
        audio = self.model.predict(arr)
        print(f"[TTS] Generated dummy wave: shape={audio.shape}")
        print("**In real TTS, we'd convert 'audio' to real sound output**")
