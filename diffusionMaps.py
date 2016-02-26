from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix, issparse, diags
from scipy.sparse.linalg import svds, eigsh
import matplotlib.pyplot as plt

class DiffusionMap(BaseEstimator):

    """
    diffusion maps for dimension reduction along the lines of [1]

    [1] Haghverdi, L., Buettner, F. & Theis, F. J.
        Diffusion maps for high-dimensional single-cell analysis of differentiation data.
        Bioinformatics 31, 2989–2998 (2015).
    """

    def __init__(self, sigma, embedding_dim, verbose=False, k=100):
        """
        :param sigma: diffusion kernel width
        :param embedding_dim: how many dimension to reduce to
        :param verbose: bool, printing some info
        :param k: kNN parameter (kNN is used when calculating the kernel matrix). the larger the more accurate, but the more RAM needed
        :return:
        """
        self.sigma = sigma
        self.embedding_dim = embedding_dim
        self.verbose = verbose
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
        if self.verbose:
            print("Calculating kernel matrix")
        kernelMat = self.__get_kernel_matrix__(X, k)
        if self.verbose:
            print("DONE: Calculating kernel matrix")

        # set the diagonal to 0: no diffusion onto itself
        kernelMat.setdiag(np.zeros(X.shape[0]))

        if density_normalize:
            if self.verbose:
                print('density normalization')
            kernelMat = self.__density_normalize__(kernelMat)

        #also, store the kernel matrix (mostly debugging)
        self.kernelMat = kernelMat

        # calculate the eigenvectors of the matrx
        if self.verbose:
            print("Calculating eigenvalue decomposition")

        lambdas, V = eigsh(kernelMat, k=self.embedding_dim)  # calculate as many eigs as the requested embedding dim

        # eigsh returns the k largest eigenvals but ascending order (smallest first), so resort
        ix = lambdas.argsort()[::-1]

        return V[:,ix], lambdas[ix]

    def __density_normalize__(self, kernelMat):
        """
        density normalization: Eq (4-5) of [1]

        be very careful here. K is a sparse matrix, which behaves differently from usual np.ndarray
        in terms of operators *, /
        """
        assert issparse(kernelMat), 'K must be sparse, multiplication behaves very differnently for sparse/dense (elementwise vs mat-mult)'

        "calculate:  P_xy / Z(x)Z(y)"
        # rescale each column by Z and also each row by Z
        # easily done by just multipling with a diagonal matrix from the left (scaling rows) and right (rescaling columsn)
        # not that row and column sum are the same as the matrix is symmetric!!

        # Z(x), kind of the partition function
        Z = np.array(kernelMat.sum(0)).flatten()  # a bit ugly, Z is this strange type(matrix), which one cannot cast into a 1d array, hence the detour to np.array
        scalingMat = diags(1.0 / Z)  # multiplying by this (one the right) is equivalent of dividing each row by Z
        P_tilde = scalingMat * kernelMat * scalingMat  # this is matrix multiply!

        # Eq (5,6) of [1]
        # once again, the same trick with diagonal matrix for resacling
        Z_tilde = np.array(P_tilde.sum(0)).flatten()
        scalingMat = diags(1.0 / Z_tilde)
        P_tilde = P_tilde * scalingMat

        return P_tilde

    def __get_NN__(self, dataMatrix, k):
        """
        caluclates the distance to the k-nearest neighbours, 
        return an array of distances and indices of nearest 
        neigbours (see NearestNeighbors.kneighbors output)

        :param dataMatrix: matrix containing one sample per row
        :param k: number of nearest nneighbours
        :return: 
        """
        if self.verbose:
            print('Calculating nearest neighbours')

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(dataMatrix)
        distances, indices = nbrs.kneighbors(dataMatrix)

        if self.verbose:
            print('DONE: Calculating nearest neighbours')
        return distances, indices

    def __get_kernel_matrix__(self, X, k):
        """
        returns the kernel matrix for the samples in X using a Gaussian Kernel and a kNN-approximation,

        - all distances are zero, except within the neartest neighbours
        - also symmetrizing the matrix (kNN is not symmetric necceseraly)

        :param X: data matrix NxF, where N=number of samples, F= number of features
        :param k: number of nearest neighbours to consider in kNN
        :return: symmetric sparse matrix of NxN
        """

        distances, indices = self.__get_NN__(X, k=k)
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