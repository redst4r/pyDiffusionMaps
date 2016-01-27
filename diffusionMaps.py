from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds,eigsh

class DiffusionMap(BaseEstimator):

    """"""

    def __init__(self, sigma, embedding_dim, verbose=False ):
        """Constructor for DiffusionMap"""
        self.sigma = sigma
        self.embedding_dim = embedding_dim
        self.verbose = verbose

    def fit_transform(self, X, kernelMatrix=False):

        k = min(50, X.shape[0])
        if not kernelMatrix:
            # calculate the kernelmatrix based on a neirest neighbour graph
            if self.verbose:
                print("Calculating kernel matrix")
            kernelMat = self.__get_kernel_matrix__(X, k)
        else:
            kernelMat = X
            assert X.shape[0] == X.shape[1], 'kernel matrix must be square'

        # set the diagonal to 0: no diffusion onto itself
        kernelMat.setdiag(np.zeros(X.shape[0]))

        # calculate the eigenvectors of the matrx
        ## U,S,Vt = svds(kernelMat)
        if self.verbose:
            print("Calculating eigenvalue decomposition")

        lambdas, V = eigsh(kernelMat,k=self.embedding_dim) # calculate as many eigs as the requested embedding dim

        return V

    def __get_NN__(self, dataMatrix, k):
        """
        caluclates the distance to the k-nearest neighbours, return an array of distances and indices of nearest neigbours (see NearestNeighbors.kneighbors output)
        :param dataMatrix:
        :return:
        """
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dataMatrix)
        distances, indices = nbrs.kneighbors(dataMatrix)
        return distances, indices
        #adj = nbrs.kneighbors_graph(dataMatrix,mode='distance') # this returns a sparse matrix, with euclidean dist of nns
        # return adj

    def __get_kernel_matrix__(self, X, k):
        distances, indices = self.__get_NN__(X, k=k)
        diffDist = np.exp(-(0.5/self.sigma**2 )* distances**2)

        # build a sparse matrix out of the diffusionDistances
        # some crazy stuff with the sparse matrixes
        N = X.shape[0]
        indptr = range(0, (N+1)*k, k)   # some helper matrix, specfiing that the first k indices in indices,flatten() belong to the first row of data

        K = csr_matrix((diffDist.flatten(), indices.flatten(), indptr), shape=(N, N))

        return K


if __name__ == '__main__':
    X = np.random.normal(0,1,size=(1000,10))
    X = np.vstack((X, np.random.normal(5,1,size=(1000,10))))


    X, y, X_test, y_test = load_mnist()

    X,y = X[:1000,:], y[:1000]
    df = DiffusionMap(sigma=0.1, embedding_dim=10)
    V = df.fit_transform(X)

    scatter(V[:,0], V[:,1], c=y)
