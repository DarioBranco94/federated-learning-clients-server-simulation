import sys
import os
import importlib.util
import tensorflow.keras as keras
import argparse
import numpy as np
import tensorflow_federated as tff
import json

class model1():
    def get_skeleton_model(self, shape) -> keras.Model:
        return keras.models.Sequential([
            keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=shape),
            keras.layers.AvgPool1D(strides=2),
            keras.layers.Conv1D(filters=48, kernel_size=5, padding='valid', activation='relu'),
            keras.layers.AvgPool1D(strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(160, activation='relu'),
            keras.layers.Dense(84, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])