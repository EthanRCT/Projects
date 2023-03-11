# condition_stability.py
"""Volume 1: Conditioning and Stability.
Ethan Crawford
Math 347
2/6/23
"""

import numpy as np
import sympy as sy
import scipy.linalg as la
import matplotlib.pyplot as plt

# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    # Find the eigenvalues of A
    _, vals, _ = np.linalg.svd(A)

    # Check for 0 eigenvalue being the lowest
    if vals[-1] == 0:
        return np.inf
    else:
        return vals[0]/vals[-1]

# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    abs_cond = []
    rel_cond = []

    for i in range(100):
        # Perturb the coefficients of the Wilkinson Polynomial slightly
        r = np.random.normal(1, 1e-10, 21)
        c_coeffs = w_coeffs*r

        # Get and sort perturbed roots
        c_roots = np.sort(np.roots(np.poly1d(c_coeffs)))
        new_roots = np.sort(np.roots(np.poly1d(w_coeffs)))

        # Plot perturbed roots
        if i == 99:
            plt.scatter(c_roots.real, c_roots.imag, marker='.', label="Perturbed", color="black", alpha=.9)
        else:
            plt.scatter(c_roots.real, c_roots.imag, marker='.', color='black', alpha=.9)
        
        # Find relative and absolute condition number
        a = np.linalg.norm((new_roots - c_roots), np.inf)/np.linalg.norm(r, np.inf)
        abs_cond.append(a)
        rel_cond.append(a*np.linalg.norm((w_coeffs), np.inf)/np.linalg.norm(new_roots, np.inf))
        
    # Plot original function
    new_roots = np.sort(np.roots(np.poly1d(w_coeffs)))
    plt.scatter(new_roots.real, new_roots.imag, label="Unperturbed", color="blue", s=30)
    plt.title("Problem 2 Plot")
    plt.legend()
    plt.show()

    return np.mean(abs_cond), np.mean(rel_cond)

# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    # construct a matrix with complex entries where the real 
    # and imaginary parts are drawn from normal distributions
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    # Compute eigenvalues and reorder
    eigA = la.eigvals(A)
    eigAH = la.eigvals(A+H)
    eigAH = reorder_eigvals(eigA, eigAH)

    # Compute absolute and relative condition numbers
    abs_cond = la.norm((eigA - eigAH), ord=2)/la.norm(H, ord=2)
    rel_cond = abs_cond*la.norm(A, ord=2)/la.norm(eigA, ord=2)

    return abs_cond, rel_cond


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    # Create linspaces
    x = np.linspace(domain[0], domain[1], res)
    y = np.linspace(domain[2], domain[3], res)

    # Init matrix
    vals = np.empty((res,res), dtype=np.float64)

    # Populate matrix
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            matrix = np.array([[1, xi], [yi, 1]], dtype=np.float64)
            _, rel = eig_cond(matrix)

            vals[i, j] = rel
    
    plt.pcolormesh(x, y, vals, cmap="gray_r")
    plt.title("Problem 4 Plot")
    plt.show()

# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    # Load data
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)

    # Solve using la.inv() for normal equations.
    x_vals1 = la.inv(A.T @ A) @ A.T @ yk

    # Solve using la.qr
    q, r = la.qr(A, mode='economic')
    x_vals2 = la.solve_triangular(r, q.T@yk)

    # Plot
    plt.plot(xk, np.polyval(x_vals1, xk), color="red", label="Inverse")
    plt.plot(xk, np.polyval(x_vals2, xk), color="blue", label="QR")
    plt.scatter(xk, yk, label="Data vals", color="black")
    plt.legend()
    plt.title("Problem 5 Plot")
    plt.show()

    return la.norm((A@x_vals1 - yk), ord=2), la.norm((A@x_vals2 - yk), ord=2)


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    # Create the list of n
    n = np.arange(5, 51, 5)

    # Init vars
    rel_forward_error = []
    x, N = sy.symbols('x, N')
    integral = x**N * sy.exp(x - 1)
    sub = (-1)**N*(sy.subfactorial(N) - sy.factorial(N)/np.e)

    # For each n
    for i in n:
        # Compute integral
        int_val = integral.subs({N:i})
        result = float(sy.integrate(int_val, (x, 0, 1)))

        # Compute the subfactorial way, and compute relative forward error
        sub_res = sub.subs({N:i})
        rel_for = np.abs(sub_res - result)/np.abs(result)
        rel_forward_error.append(rel_for)
    
    # Plot
    plt.plot(n, rel_forward_error)
    plt.yscale("log")
    plt.title("Problem 6 Plot")
    plt.show()

if __name__ == '__main__':
    '''Problem 1 Tests'''
    # A = np.array([[1, 1], [1, 1+1e-10]])
    # test = matrix_cond(A)

    '''Problem 2 Tests'''
    #print(prob2())

    '''Problem 3 Tests'''
    # A = np.random.random((4,4))
    # print(eig_cond(A))

    '''Problem 4 Tests'''
    #prob4()

    #prob6()
    pass
