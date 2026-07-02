import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	matrix = np.array(matrix)
	# Calculate the determinant of matrix
	det_matrix = np.ceil(np.linalg.det(matrix))
	# Calculat the trace of the matrix
	trace_matrix = matrix[0][0] + matrix[1][1]
	# Define and solve the quadratic equation
	coefficients = [1.0, -1.0*trace_matrix, det_matrix]
	return np.roots(coefficients)