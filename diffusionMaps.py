from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix, issparse, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy.spatial.distance
import logging
logging.basicConfig(level=logging.DEBUG)  #for DEVEL, just use full logging

def _check_Z_for_division(Z,eps):
    "Z might be zero sometimes, we add a small constant epsilon to it. make sure this doesnt change to much"

    # check that all nonzeros of z are significantly larger than eps
    ixNonZero = Z!=0
    assert np.all(np.abs(Z[ixNonZero]) > eps*10), 'values to small. might introduce some error since close to zero division'


def _density_normalize(kernelMat):
    """
    density normalization: Eq (4-5) of [1]

    be very careful here if K is a sparse matrix, which behaves differently from usual np.ndarray
    in terms of operators *, /
    """
    eps = 1e-100

    # the method only works on symmetric matrices (relies that Z is the same along rows and cols)
    atol_symmetric = 1e-10  #TODO loose tolerance
    if issparse(kernelMat):
        np.testing.assert_allclose((kernelMat-kernelMat.T).A, 0, atol=atol_symmetric)
    else:
        np.testing.assert_allclose(kernelMat-kernelMat.T, 0, atol=atol_symmetric)

    "calculate:  P_xy / Z(x)Z(y)"
    # rescale each column by Z and also each row by Z
    # easily done by just multipling with a diagonal matrix from the left (scaling rows) and right (rescaling columsn)
    # note that row and column sum are the same as the matrix is symmetric!!
    # Z(x), kind of the partition function
    if issparse(kernelMat):
        Z = np.array(kernelMat.sum(0)).flatten()  # a bit ugly, Z is this strange type(matrix), which one cannot cast into a 1d array, hence the detour to np.array
        _check_Z_for_division(Z, eps)
        scalingMat = diags(1.0 / (Z + eps), offsets=0)  # multiplying by this (one the right) is equivalent to rescaling the columns
        P_tilde = scalingMat * kernelMat * scalingMat  # this is matrix multiply!

    else:
        Z = kernelMat.sum(0).flatten() # make sure it doesnt have two dimensions, needed for the broadcasting below
        _check_Z_for_division(Z, eps)
        invZ = 1.0 / (Z + eps)  # careful about zero division.
        #TODO replace by matrix multiplicaition?!  ->  M@N
        P_tilde = kernelMat * invZ * invZ.reshape(-1,1) # broadcasts along rows and columsn, sclaing them both


    # Eq (5,6) of [1]
    # once again, the same trick with diagonal matrix for resacling
    if issparse(kernelMat):
        Z_tilde = np.array(P_tilde.sum(1)).flatten()
        _check_Z_for_division(Z_tilde, eps)
        scalingMat = diags(1.0 / (Z_tilde + eps), offsets=0)
        P_tilde = scalingMat * P_tilde
    else:
        Z_tilde = P_tilde.sum(1).flatten() # make sure it doesnt have two dimensions, needed for the broadcasting below
        _check_Z_for_division(Z_tilde, eps)
        invZ_tilde = 1.0 / (Z_tilde + eps)
        ixnonZero = Z_tilde != 0         #same fuzz about the zero

        # nasty: since zInv_tilde is a 1D vector it automatically broadcasts along rows (leading to col normalization)
        # hence we have to make the broadcasting explicit, giving shape to invZ
        P_tilde[np.ix_(ixnonZero, ixnonZero)] = P_tilde[np.ix_(ixnonZero, ixnonZero)] * invZ_tilde[ixnonZero].reshape(-1,1)  #normalizes each row

    return P_tilde


class DiffusionMap(BaseEstimator):

    """
    diffusion maps for dimension reduction along the lines of [1]
    this one uses nearest neighbours to approximate the kernel matrix

    [1] Haghverdi, L., Buettner, F. & Theis, F. J.
        Diffusion maps for high-dimensional single-cell analysis of differentiation data.
        Bioinformatics 31, 2989–2998 (2015).
    """

    def __init__(self, sigma, embedding_dim, k=100):
        """
        :param sigma: diffusion kernel width
        :param embedding_dim: how many dimension to reduce to
        :param k: kNN parameter (kNN is used when calculating the kernel matrix). the larger the more accurate, but the more RAM needed
        :return:
        """
        self.sigma = sigma
        self.embedding_dim = embedding_dim
        self.k = k

    def fit_transform(self, X, density_normalize=True):
        """
        estimates the diffusion map embedding
        :param X: data matrix (samples x features)
        :param density_normalize: boolean, wheter to apply density normalization of the kernel matrix
        :return:
        """

        # calculate the kernelmatrix based on a neirest neighbour graph
        # kernelMat is called $P_xy$ in [1]
        k = min(self.k, X.shape[0])  # limit the number of neighbours

        logging.info("Calculating kernel matrix")
        kernelMat = self._get_kernel_matrix(X, k)

        # set the diagonal to 0: no diffusion onto itself
        kernelMat.setdiag(np.zeros(X.shape[0]))  # TODO not sure if this is to be done BEFORE or after normailzation

        if density_normalize:
            logging.info("density normalization")
            kernelMat = _density_normalize(kernelMat)

        #also, store the kernel matrix (mostly debugging)
        self.kernelMat = kernelMat

        # calculate the eigenvectors of the matrx
        logging.info("Calculating eigenvalue decomposition")

        #TODO Warning: kernel matrix os not symmetric after density normalization, eigsh might fail!?
        lambdas, V = eigsh(kernelMat, k=self.embedding_dim)  # calculate as many eigs as the requested embedding dim

        # eigsh returns the k largest eigenvals but ascending order (smallest first), so resort
        ix = lambdas.argsort()[::-1]

        return V[:,ix], lambdas[ix]

    def _get_NN(self, dataMatrix, k):
        """
        caluclates the distance to the k-nearest neighbours, 
        return an array of distances and indices of nearest 
        neigbours (see NearestNeighbors.kneighbors output)

        :param dataMatrix: matrix containing one sample per row
        :param k: number of nearest nneighbours
        :return: 
        """
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(dataMatrix)
        distances, indices = nbrs.kneighbors(dataMatrix)
        return distances, indices

    def _get_kernel_matrix(self, X, k):
        """
        returns the kernel matrix for the samples in X using a Gaussian Kernel and a kNN-approximation,
        called K(x,y) in [2]

        - all distances are zero, except within the neartest neighbours
        - also symmetrizing the matrix (kNN is not symmetric necceseraly)

        :param X: data matrix NxF, where N=number of samples, F= number of features
        :param k: number of nearest neighbours to consider in kNN
        :return: symmetric sparse matrix of NxN
        """

        distances, indices = self._get_NN(X, k=k)
        diffDist = np.exp(-(0.5/self.sigma**2) * distances**2)

        # build a sparse matrix out of the diffusionDistances; some crazy magic with the sparse matrixes
        N = X.shape[0]
        indptr = range(0, (N+1)*k, k)   # some helper matrix, specfiing that the first k indices in indices,flatten() belong to the first row of data

        K = csr_matrix((diffDist.flatten(), indices.flatten(), indptr), shape=(N, N))

        # due to the kNN approximation, the matrix K is not neccesarily symmetric
        # (if x is a kNN of y, y doesnt have to be a kNN of x)
        # lets make it symmetric again, just filling in the missing entries

        shared_mask = (K!=0).multiply(K.T!=0)  # marking entries that are nonzero in both matrixes. mulitply is elemntwise!
        K_sym = K + K.T - K.multiply(shared_mask) # adding them up, subtracting the common part that was counted twice!

        np.testing.assert_allclose((K_sym-K_sym.T).A, 0, atol=1e-10)  # todo loose tolerance

        return K_sym


if __name__ == '__main__':

    # testing with MNIST
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data.astype(np.float32), mnist.target
    ix_perm = np.random.permutation(X.shape[0]) # shuffle the data
    X, y = X[ix_perm,:], y[ix_perm]

    X,y = X[:1000,:], y[:1000]

    X/=255

    df = DiffusionMap(sigma=5, embedding_dim=10)
    V,lam = df.fit_transform(X, density_normalize=False)

    plt.scatter(V[:,0], V[:,1], c=y)
    plt.show()