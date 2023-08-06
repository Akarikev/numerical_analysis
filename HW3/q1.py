import numpy as np

def lu_decomposition(A):
    """Perform LU decomposition on the given matrix A."""
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return L, U

def forward_substitution(L, b):
    """Perform forward substitution to solve the lower triangular system Lx = b."""
    n = L.shape[0]
    x = np.zeros(n)

    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x

def backward_substitution(U, y):
    """Perform backward substitution to solve the upper triangular system Ux = y."""
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

def solve_normal_equation(A, b):
    """
    Return the solution x to the linear least squares problem
    Ax ≈ b using normal equations,
    where A is an (m x n) matrix, with m > n, rank(A) = n, and
    b is a vector of size (m)
    """
    # Compute the normal equation: A.T * A * x = A.T * b
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)

    # LU decomposition
    L, U = lu_decomposition(ATA)

    # Solve the lower triangular system L * y = ATb (forward substitution)
    y = forward_substitution(L, ATb)

    # Solve the upper triangular system U * x = y (backward substitution)
    x = backward_substitution(U, y)

    return x

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

A = irisA.T  # transpose
b = irisb

# Solve the linear least squares problem
iris_x = solve_normal_equation(A, b)

# Calculate the 2-norm of the residual
residual = np.linalg.norm(np.dot(A, iris_x) - b, ord=2)

# Display the results
print("Solution x:")
print(iris_x)
print("2-norm of the residual:")
print(residual)

# Store the results in the given variables
iris_x = iris_x
iris_residual = residual
