"""Volume 2: Simplex

Ethan Crawford
3/2/23
Math 323
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        # Raise the error
        if min(b) < 0:
            raise ValueError("The given system is infeasible at the origin.")

        # Init dictionary
        self._generatedictionary(c, A, b)

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        # Add slack variables
        m = len(b)
        Abar = np.concatenate((A, np.eye(m)), axis=1)
        cbar = np.concatenate((c, np.zeros_like(b)), axis=0)
        
        # Set up dictionary
        b_Abar = np.concatenate((np.array([b]).T, -Abar), axis=1)
        zero_cbar = np.insert(cbar, 0, 0)
        D = np.concatenate((np.array([zero_cbar]), b_Abar), axis=0)
        
        self.dictionary = D

    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        # Search from left to right along the top row of the
        # dictionary (ignoring the first column), and stop 
        # once the first negative value is encountered
        top_row = self.dictionary[0]
        for index, value in enumerate(top_row[1:]):
            if value < 0:
                return index+1
        
    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        # Get pivot column and the objective function
        pivot_column = self.dictionary[1:,index]
        obj_func = -self.dictionary[1:,0]

        # Check to see if the problem is unbounded
        if np.min(pivot_column) > 0:
            raise ValueError("The problem is unbounded and has no solution")
        
        # Calculate ratios to find minimizing element
        ratios = obj_func / (pivot_column + 1e-16)

        # Get rid of negative ratios
        for index, value in enumerate(ratios):
            if value < 0:
                ratios[index] = np.inf
    
        # If a tie occurs, np.argmin() returns the first
        # element, following Bland's Rule. (+1 is because
        # we didn't include the first element at the beginning)

        return np.argmin(ratios)+1

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        piv_col = self._pivot_col()

        # Get the pivot entry
        row_index = self._pivot_row(piv_col)
        piv_entry = self.dictionary[row_index, piv_col]

        # Divide the pivot row by the negative value of the pivot entry
        self.dictionary[row_index] /= -piv_entry

        # Use the pivot row to zero out all entries in the pivot column 
        # above and below the pivot entry
        for row in range(self.dictionary.shape[0]):
            if row != row_index:
                self.dictionary[row] += self.dictionary[row_index] * self.dictionary[row, piv_col]

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        # Pivot, checking whether all of the entries in the top row of the
        # dictionary (ignoring the entry in the first column) 
        # are nonnegative
        while np.min(self.dictionary[0][1:]) < 0:
            self.pivot()

        # Initialize return dictionaries
        optimal_val = self.dictionary[0][0]
        dep_vars = {}
        indep_vars = {}

        # Calcluate dependent and independent variables
        for index, val in enumerate(self.dictionary[0][1:]):
            if val == 0: # Dependent
                # Find the index of the negative
                neg_index = np.argmin(self.dictionary[:,index+1])

                # get the value of the index in the first col
                dep_vars[index] = self.dictionary[:,0][neg_index]

            else: # Independent
               indep_vars[index] = 0

        return optimal_val, dep_vars, indep_vars

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    # Load the data
    data = np.load(filename)

    # Get the constraints and functions
    A_ = data['A']
    p = data['p']
    m = data['m']
    d = data['d']

    # Set up the problem
    c = -p
    A = np.concatenate((A_, np.eye(len(A_[0]))))
    b = np.concatenate((m, d))

    # Solve the problem
    sol = SimplexSolver(c, A, b).solve()[1]

    # Convert the dictionary into an array
    sol_array = []
    
    for _, val in sol.items():
        sol_array.append(val)

    # Return the first four elements
    return np.array(sol_array)[:4]

if __name__ == '__main__':
    A = np.array([[1,-1],[3,1],[4,3]])
    c = np.array([-3,-2])
    b = np.array([2,5,7])
    test = SimplexSolver(c,A,b)
    print(test.solve())
    #print(prob6())