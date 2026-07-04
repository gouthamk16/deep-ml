import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	output = np.zeros(len(b))
	for _ in range(n):
		output_new = np.zeros_like(output)
		for i in range(output.shape[0]):
			inter_sum = 0.0
			for j in range(output.shape[0]):
				if i!=j:
					inter_sum += A[i][j] * output[j]
			output_new[i] = (1/A[i][i]) * (b[i] - inter_sum)
		output = output_new
	return np.round(output, 4)
