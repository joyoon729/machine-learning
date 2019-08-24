import os
import pickle

import numpy as np

class TextLoader():
    def __init__(self, src_path):
        self.vocab = None
        self.src_path = src_path
        self.preprocess()

    def preprocess(self):
        with open(self.src_path, 'r') as fp:
            data = fp.read()
            # print(data)

    def load_preprocessed(self, load_path):
        data = np.load(load_path)
        print(data.shape)



loader = TextLoader('data/sample.txt').load_preprocessed('data/tinyshakespeare/data.npy')