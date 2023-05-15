from collections import Counter
import sys
import torch
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix


class LabelBalancedSampler:

    def __init__(self, A: np.array, labels: np.array, train_mask: np.array):
        self.A = A
        self.n = A.shape[0]
        self.train_mask = train_mask
        self.train_n = train_mask.sum()
        self.D = self._calculate_D()
        self.A_hat = self._calculate_A_hat()
        self.train_labels = labels[train_mask]
        # self.train_index = np.where(train_mask)[0]
        count_freq = Counter(self.train_labels.tolist())
        self.label_frequency = np.zeros((max(count_freq.keys()) + 1), dtype=int)
        for i in count_freq: self.label_frequency[i] = count_freq[i]

    def _calculate_D(self) -> np.array:
        row = np.arange(0, self.n)
        col = np.arange(0, self.n)
        data = np.asarray(self.A.sum(axis=1)).flatten().astype(float)
        data[np.where(data == 0)] = 0.001
        D = csr_matrix((data, (row, col)), shape = (self.n, self.n)).tocoo()
        return D

    def _calculate_A_hat(self) -> np.array:
        row = np.arange(0, self.n)
        col = np.arange(0, self.n)
        data = 1 / np.sqrt(self.D.data)
        D_sqrt_inverse = csr_matrix((data, (row, col)), shape = (self.n, self.n)).tocoo()
        A_hat = (D_sqrt_inverse @ self.A @ D_sqrt_inverse).tocoo()
        return A_hat

    def _node_label_frequency(self):
        return self.label_frequency[self.train_labels[np.arange(self.train_n, dtype=int)]]

    # def calculate_P(self, node_idx: int) -> float:
    #     prob = np.linalg.norm(self.A_hat.getcol(node_idx).data, ord=2) / self._node_label_frequency(node_idx)
    
    def all_probabilities(self):
        col = self.A_hat.col
        data = self.A_hat.data ** 2
        
        final = np.zeros((self.n,))
        for a, b in zip(col, data): final[a] += b
        p = self._node_label_frequency()
        prob = np.sqrt(final)[self.train_mask == 1] / p
        return prob / prob.sum()