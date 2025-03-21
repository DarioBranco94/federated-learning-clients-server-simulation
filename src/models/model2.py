import sys
import os
import importlib.util
import tensorflow.keras as keras
import argparse
import numpy as np
import tensorflow_federated as tff
import json
from keras.src.regularizers import l1

class model2():
    def get_skeleton_model(self, shape) -> keras.Model:
        return keras.models.Sequential([
            keras.layers.Conv1D(16, 5, padding='same', activation='relu', kernel_regularizer=l1(),
                   input_shape=shape),
            keras.layers.Conv1D(32, 3, padding='same', activation='relu', kernel_regularizer=l1()),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=l1()),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])