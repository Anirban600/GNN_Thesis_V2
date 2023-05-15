from collections import Counter
import sys
import torch
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix


class LabelBalancedSampler:

    def __init__(self, A: np.array, labels: np.array):
        """Label Balanced Sampler object to pick phase.
        Args:
            A (np.array): 2D array graph adjacency matrix.
            labels (np.array): 1D array label vector for each node in the graph.
        """

        self.A = A
        self.n = A.shape[0]
        self.D = self._calculate_D()
        self.A_hat = self._calculate_A_hat()

        self.labels = labels.tolist()
        self.labels_frequency = Counter(self.labels)

    def _calculate_D(self) -> np.array:
        """Calculates D, which is a diagonal matrix with degree of each node as its element.
        Returns:
            np.array: Diagonal matrix of the graph.
        """
        
        row = np.arange(0, self.n)
        col = np.arange(0, self.n)
        data = np.asarray(self.A.sum(axis=1)).flatten().astype(float)
        # self.zero_in_deg_mask = (data == 0)
        data[np.where(data == 0)] = 0.001
        D = csr_matrix((data, (row, col)), shape = (self.n, self.n)).tocoo()
        return D

    def _calculate_A_hat(self) -> np.array:
        """Calculated A_hat, which is the normalized adjancency matrix.
        Returns:
            np.array: A_hat matrix.
        """
        
        row = np.arange(0, self.n)
        col = np.arange(0, self.n)
        data = 1 / np.sqrt(self.D.data)
        D_sqrt_inverse = csr_matrix((data, (row, col)), shape = (self.n, self.n)).tocoo()
        A_hat = (D_sqrt_inverse @ self.A @ D_sqrt_inverse).tocoo()
        return A_hat

    def _node_label_frequency(self):
        node_label_count = np.array([self.labels_frequency[self.labels[i]] for i in range(self.n)])
        return node_label_count

    def calculate_P(self, node_idx: int) -> float:
        prob = np.linalg.norm(self.A_hat.getcol(node_idx).data, ord=2) / self._node_label_frequency(node_idx)
    
    def all_probabilities(self):
        col = self.A_hat.col
        data = self.A_hat.data ** 2
        
        final = np.zeros((self.n,))
        for a, b in zip(col, data): final[a] += b
        prob = np.sqrt(final) / self._node_label_frequency()
        return torch.tensor(prob)