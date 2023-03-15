# image_segmentation.py
"""Volume 1: Image Segmentation.
Ethan Crawford Taylor
Math 345
10/31/22
"""

import numpy as np
from scipy import linalg as la
from imageio.v2 import imread
from matplotlib import pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    return(np.diag(np.sum(A, axis=1)) - A)

# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    # Get the eigenvalues of the laplacian and filter for zero eigenvalues
    L = np.sort(np.real(la.eigvals(laplacian(A))))
    return np.where((np.abs(L)<tol) & (np.abs(L)>=0))[0].size, L[1]

# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(int), R[mask]

# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)
        self.image = image / 255
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        # If the len of the shape of the image is 3, it is color (RGB channels)
        if len(self.image.shape) == 3:
            self.brightness = np.ravel(self.image.mean(axis=2))
        else:
            self.brightness = np.ravel(self.image)
            
    # Problem 3
    def show_original(self):
        """Display the original image."""
        size = len(self.image.shape)

        # Print the grayscale image
        if size < 3:
            cmap = "gray"
        else:
            cmap = None

        plt.imshow(self.image, cmap=cmap)
        plt.axis("off")
        plt.show()      

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        # Compute the adjacency matrix A and degree matrix D
        mn = len(self.brightness)
        A = scipy.sparse.lil_matrix((mn,mn))
        D = np.empty(mn)
        
        # Calculate the weights for each pixel
        for i in range(mn):
            neighbors,distances = get_neighbors(i,r,self.height,self.width)
            weights = np.exp(-(np.abs(self.brightness[i]-self.brightness[neighbors])/sigma_B2)-(np.abs(distances)/sigma_X2))
            A[i, neighbors] = weights
            D[i] = np.sum(weights)

        # Convert A to a sparse matrix and return
        A = A.tocsc()    
        return A,D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        # Compute the Laplacian
        L = scipy.sparse.csgraph.laplacian(A)

        # Construct a sparse diagonal matrix
        sp = scipy.sparse.diags(1/np.sqrt(D)).tocsc()
        Dhalf = sp@L@sp     

        # Use scipy.sparse.linalg.eigsh() to compute the eigenvector corresponding to the
        # second-smallest eigenvalue of D L D
        _,vec = scipy.sparse.linalg.eigsh(Dhalf, which="SM", k=2)

        # Reshape the eigenvector as a m × n matrix and use this matrix to construct the desired
        # boolean mask
        vec = vec[:,1].reshape(self.height,self.width)
        mask = vec > 0

        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        # Compute the adjacency and degree matrices
        # Compute the boolean mask that segments the image
        A, D = self.adjacency()
        mask = self.cut(A, D)
        
        # Display the original image and its segments
        size = len(self.image.shape)
        if size < 3:
            updated_mask = mask
            cmap = "gray"
        elif size == 3:
            updated_mask = np.dstack((mask,mask,mask))
            cmap = None

        (ax1, ax2, ax3) = plt.subplots(1, 3)[1]
        ax1.imshow(self.image * updated_mask, cmap=cmap)
        ax2.imshow(self.image * ~updated_mask, cmap=cmap)
        ax3.imshow(self.image, cmap=cmap)

        plt.tight_layout()

        if cmap == "gray":
            plt.suptitle("Black and White", y=.8)
        else:
            plt.suptitle("Color", y=.8)

        for ax, title in zip((ax1, ax2, ax3), ("Segmented", "Segmented", "Original")):
            ax.axis('off')
            ax.set_title(title)

        plt.gcf().set_dpi(150)
        plt.tight_layout()
        plt.show()



