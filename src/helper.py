import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, TypeVar, List, Tuple
import logging
from random import randrange
from logzero import setup_logger

from config import CONFIG

T = TypeVar("Generic Data")
X = TypeVar("Input Data")
y = TypeVar("Input Target")
Vector = List[T]

logger = setup_logger(__file__, level=logging.DEBUG)

def perf_measure(y_actual, y_hat, pos=1):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==pos:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==(pos - 1):
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return (precision, recall)

def metric(y, y_hat):
	metrics_1 = perf_measure(y, y_hat, pos=1)
	metrics_2 = perf_measure(y, y_hat, pos=0)
	print("\t| Precision \t| Recall \t|")
	print("-"*50)
	print(f"0\t| {metrics_2[0]:.3f} \t| {metrics_2[1]:.3f} \t|")
	print(f"1\t| {metrics_1[0]:.3f} \t| {metrics_1[1]:.3f} \t|")
	print("-"*50)

def one_hot_encode(y: y) -> Tuple[List[List[float]], List[X]]:
	K = np.unique(y)
	one_hot = np.zeros((len(y), len(K)))
	for idx, label in enumerate(K):
		one_hot[y==label, idx] = 1
	columns = K
	one_hot = pd.DataFrame(one_hot, columns=columns)
	return one_hot, K

class kfoldCrossVal:
	"""Kfold Cross Validation
	"""
	def __init__(self, k: int) -> None:
		self.k: int = k
	def fit(self, df: Iterable[T]) -> None:
		"""Randome generate a kfold scheme for cross validation

		Args:
			df (Iterable[T]): input data
		"""
		self.N: int = len(df)
		self.train_indices: List[List[int]] = []
		self.test_indices: List[List[int]] = []
		self.ksize: int = int(self.N / self.k)
		self.df_indices: List[int] = list(range(self.N))
	
	def split(self):
		for k in range(self.k):
			fold = list()
			while len(fold) < self.ksize:
				index = randrange(len(self.df_indices))
				fold.append(self.df_indices.pop(index))
			self.test_indices.append(fold)
			self.train_indices.append([x for x in range(self.N) if x not in fold])
		return zip(self.train_indices, self.test_indices)
def approx_equal(x, y, eps=0.1):
	return abs(x - y) <= eps
def train_test_split1(X: X, y: y, size: float=0.2, stratify: bool=False) -> Tuple[Tuple[X, y], Tuple[X, y]]:
	"""split train and test sets with specified size.

	Args:
		X (X): Input data (an iterable)
		y (y): Input target (an iterable)
		size (float, optional): Propotion of train and test set. Defaults to 0.2 (split train/test = 80/20).
		stratify (bool, optional): Split train and test set according to target y distribution. Defaults to False.

	Returns:
		Tuple[X, y]: Train and test sets
	"""
	columns = None
	if isinstance(X, pd.DataFrame):
		columns = X.columns
		X = X.to_numpy()
		y = y.to_numpy()

	elif not isinstance(X, np.ndarray):
		X = np.array(X)
	N = len(X)
	# sample_size = int(np.ceil(size * N))
	sample_size = int(size * N)
	def __split():
		if not stratify:
			# randomly sample indices 
			idx = np.random.choice(N, size=sample_size, replace=False)
			# print(f"Number of samples to be taken in test set: {len(np.unique(idx))}")
			mask = np.ones(N, dtype=bool)
			mask[idx] = False
			X_train = X[mask]
			y_train = y[mask]
			X_test = X[~mask]
			y_test = X[~mask]
			assert (len(X_test) / N) == size, "Check your splitting function!"
			assert (len(X_train) / N) == (1 - size), "Check your splitting function"
		else:
			labels, counts = np.unique(y, return_counts=True)
			X_train, y_train = [], []
			X_test, y_test = [], []
			for label, count in zip(labels, counts):
				x_train, ytrain = X[y==label], y[y==label]
				sample_size = int(size * count)
				n = len(x_train)
				idx = np.random.choice(n, size=sample_size, replace=False)
				mask = np.ones(n, dtype=bool)
				mask[idx] = False
				x_train_train = x_train[mask]
				y_train_train = ytrain[mask]
				x_train_test = x_train[~mask]
				y_train_test = ytrain[~mask]
				X_train.extend(x_train_train)
				X_test.extend(x_train_test)
				y_train.extend(y_train_train)
				y_test.extend(y_train_test)
				# print(len(x_train_test) / n)  
				# print(size)
				assert approx_equal((len(x_train_test) / n), size), "Check your stratify splitting!"
			assert approx_equal((len(X_test) / N), size), "Check your splitting function!"
		return (X_train, np.squeeze(y_train)), (X_test, np.squeeze(y_test))
	(X_train, y_train), (X_test, y_test) = __split()
	if columns is not None:
		X_train = pd.DataFrame(X_train, columns=columns)
		# y_train = pd.DataFrame(np.squeeze(y_train))
		X_test = pd.DataFrame(X_test, columns=columns)
		# y_test = pd.DataFrame(np.squeeze(y_test))
	else:
		return (np.array(X_train), y_train), (np.array(X_test), y_test)

	return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
	kfold = kfoldCrossVal(k=5)
	train_data = pd.read_csv(CONFIG.data / "raw" / "FakeBank_churn.csv")
	kfold.fit(train_data)
	for train_indices, test_indices in kfold.split():
		# logger.info("Train and test indices:")
		# logger.info(f"{len(train_indices)}, {len(test_indices)}")
		whole_indices = train_indices + test_indices
		whole_indices.sort()
		assert whole_indices == list(range(kfold.N)), "wrong split!"
	assert kfold.test_indices[0] != kfold.test_indices[1], "your kfold returns the same indices for all slits!"

	(X_train, y_train), (X_test, y_test) = train_test_split(train_data.drop("Exited", axis=1), train_data.Exited, size=0.3, stratify=True)
	print(X_train.shape, X_test.shape)
    
