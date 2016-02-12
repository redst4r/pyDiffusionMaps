from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix, issparse
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

        # calculate the eigenvectors of the matrx
        if self.verbose:
            print("Calculating eigenvalue decomposition")

        lambdas, V = eigsh(kernelMat, k=self.embedding_dim)  # calculate as many eigs as the requested embedding dim

        # eigsh returns the k largest eigenvals but ascending order (smallest first), so resort
        ix = lambdas.argsort()[::-1]

        #also, store the kernel matrix (mostly debugging)
        self.kernelMat = kernelMat
        return V[:,ix], lambdas[ix]

    def __density_normalize__(self, kernelMat):
        """
        density normalization: Eq (4-5) of [1]

        be very careful here. K is a sparse matrix, which behaves differently from usual np.ndarray
        in terms of operators *, /
        """
        assert issparse(kernelMat), 'K must be sparse, multiplication behaves very differnently for sparse/dense (elementwise vs mat-mult)'

        # Z(x), kind of the partition function
        Z = kernelMat.sum(0)   # WARNING Z is of type matrix

        # here we now pretend/define that x runs along the columns of the matrix (x is the column index)
        P_tilde = kernelMat / np.dot(Z.reshape(-1,1), Z.reshape(1,-1))  # P_xy / Z(x)Y(y)

        # Eq (5,6) of [1]
        Z_tilde = P_tilde.sum(0)
        P_tilde /= Z_tilde

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
        distances, indices = self.__get_NN__(X, k=k)
        diffDist = np.exp(-(0.5/self.sigma**2) * distances**2)

        # build a sparse matrix out of the diffusionDistances
        # some crazy magic with the sparse matrixes
        N = X.shape[0]
        indptr = range(0, (N+1)*k, k)   # some helper matrix, specfiing that the first k indices in indices,flatten() belong to the first row of data

        K = csr_matrix((diffDist.flatten(), indices.flatten(), indptr), shape=(N, N))

        # due to the kNN approximation, the matrix K is not neccesarily symmetric
        # (if x is a kNN of y, y doesnt have to be a kNN of x)
        # lets make it symmetric again, just filling in the missing entries

        # TODO K + K.T*(K==0) is inefficient due to the K==0 where K is sparse
        assert issparse(K), 'K must be sparse, multiplication behaves very differnently for sparse/dense (elementwise vs mat-mult)'
        K = K + K.T.multiply((K==0)) # WARNING: for sparse matrices, '*' is overloaded as matrix multiplication!!

        return K


if __name__ == '__main__':
    #X = np.random.normal(0,1,size=(1000,10))
    #X = np.vstack((X, np.random.normal(5,1,size=(1000,10))))

    import sys
    sys.path.append('/home/michi/pythonProjects/deepLearning/utils')
    sys.path.append('/home/michi/pythonProjects/deepLearning/AE_timeLapse')
    sys.path.append('/home/michi/pythonProjects/deepLearning/pyDiffusionMaps')

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