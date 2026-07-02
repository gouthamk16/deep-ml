import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	# Check if T and S are invertible. i.e., det is non zero
	if np.isclose(np.ceil(np.linalg.det(np.array(T))), 0) or np.isclose(np.ceil(np.linalg.det(np.array(S))), 0):
		return -1
	a, t, s = np.array(A), np.array(T), np.array(S)
	t_inv = np.linalg.inv(t)
	if t_inv.shape[1]!=a.shape[0]:
		return -1
	t_inv_a = t_inv @ a
	if t_inv_a.shape[1] != s.shape[0]:
		return -1
	return t_inv_a @ s