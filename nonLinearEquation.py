import math
import matplotlib.pyplot as plt

# Function for bisection method
def bisection_method(f, a, b, epsilon, max_iterations):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge on the given interval.")
        return None, -1
    
    c = a
    f_values = []
    relative_errors = []
    for i in range(max_iterations):
        c = (a + b) / 2
        f_c = f(c)
        f_values.append(f_c)
        
        if i > 0:
            relative_error = abs((c - prev_c) / c)
            relative_errors.append(relative_error)
            if relative_error < epsilon:
                return c, 1, f_values, relative_errors
        
        if abs(f_c) < epsilon:
            return c, 2, f_values, relative_errors
        
        if f_c * f(a) < 0:
            b = c
        else:
            a = c
        
        prev_c = c
    
    print("Bisection method did not converge within the maximum number of iterations.")
    return None, 0, f_values, relative_errors

# Function for fixed-point method
def fixed_point_method(g, x0, epsilon, max_iterations):
    x = x0
    f_values = []
    relative_errors = []
    for i in range(max_iterations):
        x_new = g(x)
        f_val = abs(x_new - x)
        f_values.append(f_val)
        
        if i > 0:
            relative_error = abs((x_new - x) / x_new)
            relative_errors.append(relative_error)
            if relative_error < epsilon:
                return x_new, 1, f_values, relative_errors
        
        if f_val < epsilon:
            return x_new, 2, f_values, relative_errors
        
        x = x_new
    
    print("Fixed-point method did not converge within the maximum number of iterations.")
    return None, 0, f_values, relative_errors

# Function for Newton-Raphson method
def newton_raphson_method(f, f_prime, x0, epsilon, max_iterations):
    x = x0
    f_values = []
    relative_errors = []
    for i in range(max_iterations):
        f_val = f(x)
        f_prime_val = f_prime(x)
        f_values.append(abs(f_val))
        
        if i > 0:
            relative_error = abs((x - prev_x) / x)
            relative_errors.append(relative_error)
            if relative_error < epsilon:
                return x, 1, f_values, relative_errors
        
        if abs(f_val) < epsilon:
            return x, 2, f_values, relative_errors
        
        if f_prime_val == 0:
            print("Newton-Raphson method failed due to zero derivative.")
            return None, 0, f_values, relative_errors
        
        x -= f_val / f_prime_val
        
        prev_x = x
    
    print("Newton-Raphson method did not converge within the maximum number of iterations.")
    return None, 0, f_values, relative_errors

# Test function 1: f(x) = x - cos(x)
def equation1(x):
    return x - math.cos(x)

def equation1_derivative(x):
    return 1 + math.sin(x)

# Test function 2: f(x) = e^(-x) - x
def equation2(x):
    return math.exp(-x) - x

def equation2_derivative(x):
    return -math.exp(-x) - 1

# Test function 3: f(x) = x^4 - 7.4x^3 + 20.44x^2 - 24.184x + 9.6448
def equation3(x):
    return x**4 - 7.4*x**3 + 20.44*x**2 - 24.184*x + 9.6448

def equation3_derivative(x):
    return 4*x**3 - 22.2*x**2 + 40.88*x - 24.184

print("Root Finding Program")
print("Choose a method:")
print("(1) Bisection")
print("(2) Fixed-Point")
print("(3) Newton-Raphson")

choice = int(input("Enter your choice: "))

if choice == 1:
    f = equation1
    a = float(input("Enter the starting point 'a': "))
    b = float(input("Enter the starting point 'b': "))

    epsilon = float(input("Enter the convergence criterion for relative approximate errors: ")) / 100
    max_iterations = int(input("Enter the maximum number of iterations: "))

    root, flag, f_values, relative_errors = bisection_method(f, a, b, epsilon, max_iterations)
    if root is not None:
        print("Root found: x =", root)
        print("Stopping criteria flag:", flag)

    # Plot f(x) vs. x
    x_values = list(range(len(f_values)))
    plt.plot(x_values, f_values)
    plt.xlabel("Iteration Number")
    plt.ylabel("f(x)")
    plt.title("f(x) vs. x (Bisection Method)")
    plt.show()

    # Plot approximate relative error vs. iteration number
    x_values = list(range(len(relative_errors)))
    plt.plot(x_values, relative_errors)
    plt.xlabel("Iteration Number")
    plt.ylabel("Approximate Relative Error")
    plt.title("Approximate Relative Error vs. Iteration Number (Bisection Method)")
    plt.show()

elif choice == 2:
    g = equation1_derivative
    x0 = float(input("Enter the starting point 'x0': "))

    epsilon = float(input("Enter the convergence criterion for relative approximate errors: ")) / 100
    max_iterations = int(input("Enter the maximum number of iterations: "))

    root, flag, f_values, relative_errors = fixed_point_method(g, x0, epsilon, max_iterations)
    if root is not None:
        print("Root found: x =", root)
        print("Stopping criteria flag:", flag)

    # Plot f(x) vs. x
    x_values = list(range(len(f_values)))
    plt.plot(x_values, f_values)
    plt.xlabel("Iteration Number")
    plt.ylabel("f(x)")
    plt.title("f(x) vs. x (Fixed-Point Method)")
    plt.show()

    # Plot approximate relative error vs. iteration number
    x_values = list(range(len(relative_errors)))
    plt.plot(x_values, relative_errors)
    plt.xlabel("Iteration Number")
    plt.ylabel("Approximate Relative Error")
    plt.title("Approximate Relative Error vs. Iteration Number (Fixed-Point Method)")
    plt.show()

elif choice == 3:
    f = equation3
    f_prime = equation3_derivative
    x0 = float(input("Enter the starting point 'x0': "))

    epsilon = float(input("Enter the convergence criterion for relative approximate errors: ")) / 100
    max_iterations = int(input("Enter the maximum number of iterations: "))

    root, flag, f_values, relative_errors = newton
