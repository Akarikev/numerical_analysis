import numpy as np

def LU_decomposition(A):
    """
    LU decomposition of matrix A
    """
    m, n = A.shape
    L = np.eye(m, dtype=float)
    U = A.copy()

    for i in range(n):
        pivot = U[i, i]
        if pivot == 0:
            raise ValueError("Matrix is singular.")

        for j in range(i + 1, m):
            factor = U[j, i] / pivot
            L[j, i] = factor
            U[j, :] -= factor * U[i, :]

    return L, U

def back_substitution(U, b):
    """
    Backward substitution to solve the upper triangular system Ux = b
    """
    n = U.shape[1]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def solve_normal_equation(A, b):
    """
    Return the solution x to the linear least squares problem
    Ax ≈ b using normal equations,
    where A is an (m x n) matrix, with m > n, rank(A) = n, and
    b is a vector of size (m)
    """
    # Compute A^T A and A^T b
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)

    # LU decomposition of ATA
    L, U = LU_decomposition(ATA)

    # Solve the system Ux = ATb using backward substitution
    x = back_substitution(U, ATb)

    return x



# Previous code for solve_normal_equation function
# ...

# Given data
irisA = np.array([[4.8, 6.3, 5.7, 5.1, 7.7, 5.6, 4.9, 4.4, 6.4, 6.2, 6.7, 4.5, 6.3,
                  4.8, 5.8, 4.7, 5.4, 7.4, 6.4, 6.3, 5.1, 5.7, 6.5, 5.5, 7.2, 6.9,
                  6.8, 6., 5., 5.4, 5.6, 6.1, 5.9, 5.6, 6., 6., 4.4, 6.9, 6.7,
                  5.1, 6., 6.3, 4.6, 6.7, 5., 6.7, 5.8, 5.1, 5.2, 6.1],
                 [3.1, 2.5, 3., 3.7, 3.8, 3., 3.1, 2.9, 2.9, 2.9, 3.1, 2.3, 2.5,
                  3.4, 2.7, 3.2, 3.7, 2.8, 2.8, 3.3, 2.5, 4.4, 3., 2.6, 3.2, 3.2,
                  3., 2.9, 3.6, 3.9, 3., 2.6, 3.2, 2.8, 2.7, 2.2, 3.2, 3.1, 3.1,
                  3.8, 2.2, 2.7, 3.1, 3., 3.5, 2.5, 4., 3.5, 3.4, 3.],
                 [1.6, 4.9, 4.2, 1.5, 6.7, 4.5, 1.5, 1.4, 4.3, 4.3, 5.6, 1.3, 5.,
                  1.6, 5.1, 1.3, 1.5, 6.1, 5.6, 6., 3., 1.5, 5.8, 4.4, 6., 5.7,
                  5.5, 4.5, 1.4, 1.3, 4.1, 5.6, 4.8, 4.9, 5.1, 5., 1.3, 4.9, 4.4,
                  1.5, 4., 4.9, 1.5, 5.2, 1.3, 5.8, 1.2, 1.4, 1.4, 4.9]])
irisb = np.array([0.2, 1.5, 1.2, 0.4, 2.2, 1.5, 0.1, 0.2, 1.3, 1.3, 2.4, 0.3, 1.9,
                  0.2, 1.9, 0.2, 0.2, 1.9, 2.1, 2.5, 1.1, 0.4, 2.2, 1.2, 1.8, 2.3,
                  2.1, 1.5, 0.2, 0.4, 1.3, 1.4, 1.8, 2., 1.6, 1.5, 0.2, 1.5, 1.4,
                  0.3, 1., 1.8, 0.2, 2.3, 0.3, 1.8, 0.2, 0.2, 0.2, 1.8])

# Transpose the matrix to get 50x3
A = irisA.T  

b = irisb

# Solve the linear least squares problem
iris_x = solve_normal_equation(A, b)

# Calculate the residual
residual_vector = np.matmul(A, iris_x) - b
iris_residual = np.linalg.norm(residual_vector, ord=2)

# Print the results
print("Solution x:", iris_x)
print("2-norm of the residual:", iris_residual)

