from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix, issparse, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy.spatial.distance
import logging
logging.basicConfig(level=logging.INFO)  #for DEVEL, just use full logging


"""
    [1] Haghverdi, L., Buettner, F. & Theis, F. J.
        Diffusion maps for high-dimensional single-cell analysis of differentiation data.
        Bioinformatics 31, 2989–2998 (2015).
    [2] Laleh Haghverdi, Maren B¨uttner, F. Alexander Wolf, Florian Buettner, Fabian J. Theis
        Diffusion pseudotime robustly reconstructs lineage branching
"""

def _check_Z_for_division(Z,eps):
    "Z might be zero sometimes, we add a small constant epsilon to it. make sure this doesnt change to much"

    # check that all nonzeros of z are significantly larger than eps
    ixNonZero = Z!=0
    assert np.all(np.abs(Z[ixNonZero]) > eps*10), 'values to small. might introduce some error since close to zero division'


def _density_normalize(kernelMat, symmetrize=False):
    """
    1. density normalization: Eq (4-5) of [1]  or Eq 3,4 in [2]
        W_xy = K(x,y)/Z(x)Z(y)
        thats the Coifman anisotropy thingy, trying to mitigate the effect of density
        (alpha=1 in Coifman)

    2. strange row normalization Eq(5,6) in [2]  or Eq(5,6) in [1]
        this is to get the "normalized graph laplacian" as in Coifman.

        essentially this makes it a transition matrix. This is asymmetric!

    3. optional: symmetrize the transition matrix again! (see [2] Suppl.Eq 7)

    be very careful here if K is a sparse matrix, which behaves differently from usual np.ndarray
    in terms of operators *, /

    :param symmetrize: if True, we return a symmetrized transition matrix
         otherwise the classic non-symmetric transition matrix
    """
    eps = 1e-100

    # the method only works on symmetric matrices (relies that Z is the same along rows and cols)
    atol_symmetric = 1e-10  #TODO loose tolerance
    if issparse(kernelMat):
        np.testing.assert_allclose((kernelMat-kernelMat.T).A, 0, atol=atol_symmetric)
    else:
        np.testing.assert_allclose(kernelMat-kernelMat.T, 0, atol=atol_symmetric)

    "calculate:  P_xy / Z(x)Z(y)"
    "rescale each column by Z and also each row by Z"
    "easily done by just multipling with a diagonal matrix from the left (scaling rows) and right (rescaling columsn)"
    # note that row and column sum are the same as the matrix is symmetric!!
    if issparse(kernelMat):
        Z = np.array(kernelMat.sum(0)).flatten()  # a bit ugly, Z is this strange type(matrix), which one cannot cast into a 1d array, hence the detour to np.array
        _check_Z_for_division(Z, eps)
        scalingMat = diags(1.0 / (Z + eps), offsets=0)  # multiplying by this (one the right) is equivalent to rescaling the columns
        P_tilde = scalingMat * kernelMat * scalingMat  # this is matrix multiply!
        # assert np.testing.assert_allclose(P_tilde.toarray(), P_tilde.T.toarray(), err_msg='Ptilde should be symmetric')

    else:
        Z = kernelMat.sum(0).flatten()  # make sure it doesnt have two dimensions, needed for the broadcasting below
        _check_Z_for_division(Z, eps)
        invZ = 1.0 / (Z + eps)  # careful about zero division.
        #TODO replace by matrix multiplicaition?!  ->  M@N
        P_tilde = kernelMat * invZ * invZ.reshape(-1,1)  # broadcasts along rows and columsn, sclaing them both
        # assert np.testing.assert_allclose(P_tilde, P_tilde.T, err_msg='Ptilde should be symmetric', atol=atol_symmetric)

    "THIS PTILDE HAS TO BE SYMMETRIC HERE!!"
    logging.warning("max discrepancy of Ptilde symmetry: %e" % np.max(np.abs(P_tilde - P_tilde.T)))

    # Eq (5,6) of [1]
    # once again, the same trick with diagonal matrix for resacling
    if issparse(kernelMat):
        # import pdb
        # pdb.set_trace()
        if symmetrize:   # Eq 7 of
            logging.warning("not clear how the symmetric version is implemented")
            rowsum = np.array(P_tilde.sum(1)).flatten()
            _check_Z_for_division(rowsum, eps)
            scalingMat_rows = diags(1.0 / (rowsum + eps), offsets=0)
            sqrt_scale_row = np.sqrt(scalingMat_rows)

            colsum = np.array(P_tilde.sum(0)).flatten()
            _check_Z_for_division(colsum, eps)
            scalingMat_cols = diags(1.0 / (colsum + eps), offsets=0)
            sqrt_scale_col = np.sqrt(scalingMat_cols)

            logging.warning("max discrepancy of row/colsum: %e" % np.max(np.abs(rowsum-colsum)))
            logging.warning("max discrepancy of sqrt: %e" % np.max(np.abs(sqrt_scale_row-sqrt_scale_col)))

            P_tilde =  sqrt_scale_row * P_tilde * sqrt_scale_col
        else:
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
        if symmetrize:   # Eq 7 of
            raise NotImplementedError("not clear how symmetric is implemented. ask maren")
            logging.warning("not clear how the symmetric version is implemented")
            sqrt_invZ_tilde = np.sqrt(invZ_tilde)
            P_tilde[np.ix_(ixnonZero, ixnonZero)] = P_tilde[np.ix_(ixnonZero, ixnonZero)] * sqrt_invZ_tilde[ixnonZero].reshape(-1, 1) * sqrt_invZ_tilde[ixnonZero] # normalizes each row
        else:
            P_tilde[np.ix_(ixnonZero, ixnonZero)] = P_tilde[np.ix_(ixnonZero, ixnonZero)] * invZ_tilde[ixnonZero].reshape(-1,1)  #normalizes each row

    return P_tilde


def _calc_dpt(T):
    ":param T: transition matrix"
    n_vectors = T.shape[0]-1  # somehow the method can only compute all but the first EV

    assert issparse(T), "T should be sparse"
    logging.info("Calculating full eigenvalue decomposition")
    lambdas, V = eigsh(T, k=n_vectors)  # psi(0) which is the stationary density

    # the last eigenvalue/eigenvector pair is the stationary state which we ommit here
    # note that we're missing the smalest eigenvector here!!
    prefactor = lambdas/(1-lambdas)

    M = V[:,:-1] @ np.diag(prefactor[:-1]) @ V[:,:-1].T  # [:-1] skip the last EV which is the steady state

    logging.info("calculating dpt matrix")

    # we have to iterate over all eigenvectors,
    # build a difference matrix and multiply by the prefactor
    # dpt2 is then jsut the sum over all these matrixes
    dpt2_matrix = np.zeros((V.shape[0], V.shape[0]))
    for i in range(0, n_vectors - 1):  # -1 again to skip the stst-vector
        currentPsi = V[:, i].reshape(-1, 1)  # a row vector
        # due to numpy broadcasting the next line will
        # become a matrix: difference of everyone vs evergone
        squared_difference_matrix = (currentPsi - currentPsi.T) ** 2
        dpt2_matrix = dpt2_matrix + prefactor[i]**2 * squared_difference_matrix

    import warnings
    warnings.warn('changed to return sqrt(dtp2)')
    return M, np.sqrt(dpt2_matrix)


class DiffusionMap(BaseEstimator):

    """
    diffusion maps for dimension reduction along the lines of [1]
    this one uses nearest neighbours to approximate the kernel matrix


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
        self.local_sigma = None

        # NN is the most expensive caluclation, cache it
        self._cached_nn_distances = None
        self._cached_nn_indices = None

    def fit_transform(self, X, density_normalize=True, symmetrize=False):
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
            kernelMat = _density_normalize(kernelMat, symmetrize=symmetrize)

        #also, store the kernel matrix (mostly debugging)
        self.kernelMat = kernelMat

        # calculate the eigenvectors of the matrx
        logging.info("Calculating eigenvalue decomposition")

        #TODO Warning: kernel matrix os not symmetric after density normalization, eigsh might fail!?
        lambdas, V = eigsh(kernelMat, k=self.embedding_dim)  # calculate as many eigs as the requested embedding dim

        # eigsh returns the k largest eigenvals but ascending order (smallest first), so resort
        ix = lambdas.argsort()[::-1]

        # TODO could think about getting rid of the first EV, which has only density info
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

        if self._cached_nn_distances is None or self._cached_nn_indices is None:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(dataMatrix)
            distances, indices = nbrs.kneighbors(dataMatrix)

            # cache for later
            self._cached_nn_distances = distances
            self._cached_nn_indices = indices

        else:  # load cached
            distances = self._cached_nn_distances
            indices = self._cached_nn_indices

        return distances, indices

    def _get_kernel_matrix(self, X, k):
        """
        returns the kernel matrix for the samples in X using a Gaussian Kernel and a kNN-approximation,
        called K(x,y) in [2]

        - all distances are zero, except within the neartest neighbours
        - also symmetrizing the matrix (kNN is not symmetric necceseraly)

        if self.sigma !=0 just apply a single specified sigma to all datapoints.
        if self.sigma ==0, estimate sigma for each datapoint via nearest-neighbour distance

        :param X: data matrix NxF, where N=number of samples, F= number of features
        :param k: number of nearest neighbours to consider in kNN
        :return: symmetric sparse matrix of NxN
        """

        distances, indices = self._get_NN(X, k=k)

        if self.sigma != 0:
            logging.info("calculating kernel matrix with global sigma %f" % self.sigma)
            diffDist = np.exp(-(0.5 / self.sigma**2) * distances**2)
        else:
            logging.info("calculating kernel matrix with local sigma")
            local_sigma_squared =  np.median(distances**2, axis=1).reshape(-1,1)  # .shape = (datapoints, 1)
            local_sigma_squared += 1e-15  # numerical stability, also dropout leads to 0 distance of different datapoints

            self.local_sigma =  np.sqrt(local_sigma_squared)
            "more tricky as for each datapoint + knn,s we have to consider different sigmas"
            # distances.shape = (datapoints, kNNs)
            diffDist = []
            for i in range(len(indices)):  # for each datapoint calculate the row in the kernel matrix, taking care of the local sigmas of each datapoint

                prefactor_nom = 2 * self.local_sigma[i] * self.local_sigma[indices[i]]
                prefactor_denom = local_sigma_squared[i] + local_sigma_squared[indices[i]]
                prefactor = np.sqrt(prefactor_nom/prefactor_denom)
                exp_denom = 2 * prefactor_denom
                diffDist.append(prefactor * np.exp(-(distances[i].reshape(-1,1) ** 2)/ exp_denom))  # reshape otherwise autobroadcasting goes from (k,) -> (k,k)
            diffDist = np.array(diffDist)


        # build a sparse matrix out of the diffusionDistances; some crazy magic with the sparse matrixes
        N = X.shape[0]
        indptr = range(0, (N+1)*k, k)   # some helper matrix, specfiing that the first k indices in indices,flatten() belong to the first row of data

        K = csr_matrix((diffDist.flatten(), indices.flatten(), indptr), shape=(N, N))

        # due to the kNN approximation, the matrix K is not neccesarily symmetric
        # (if x is a kNN of y, y doesnt have to be a kNN of x)
        # lets make it symmetric again, just filling in the missing entries

        shared_mask = (K!=0).multiply(K.T!=0)  # marking entries that are nonzero in both matrixes. mulitply is elemntwise!
        K_sym = K + K.T - K.multiply(shared_mask) # adding them up, subtracting the common part that was counted twice!

        np.testing.assert_allclose((K_sym-K_sym.T).A, 0, atol=1e-6)  # todo loose tolerance

        return K_sym


class DiffusionMap_Nystroem(BaseEstimator):
    """
    this one uses the Nystroem approximation to approximate the kernel matrix
    """

    def __init__(self, sigma, embedding_dim, m_nystroem, verbose=False, ):
        """Constructor for DiffusionMap_Nystroem
        :param sigma: diffusion kernel width
        :param embedding_dim: how many dimension to reduce to
        :param verbose: bool, printing some info
        :param m_nystroem: how many samples to use when approximating the  kernel matrix (larger-> more accurate, but more memory/computation time)
        :return:
        """
        self.sigma = sigma
        self.embedding_dim = embedding_dim
        self.verbose = verbose
        self.m_nystroem = m_nystroem

    def fit_transform(self, X, weighted_nystroem=False, density_normalize=True):
        """
        along the lines of "Using the nystroem method to speed up kernel machines"
        :param X:
        :param weighted_nystroem:
        :return:
        """
        N = X.shape[0]
        if weighted_nystroem:
            raise NotImplementedError('TODO weighted nystroem')
        else:
            # draw a few (m) samples on which the kernel matrix is actually calculated
            ix = np.random.choice(N, size=self.m_nystroem, replace=False)

        self.ix_subsample = ix  # for debugging mostly

        # calculate the small kernel matrix
        if self.verbose: print('calculating small kernel matrix')
        K_mm = self._calc_kernel_mat(X[ix,:])

        # set the diagonal to 0: no diffusion onto itself
        np.fill_diagonal(K_mm, 0)
        if density_normalize:
            K_mm = _density_normalize(K_mm)

        # save (for debugging mostly)
        self.K_mm = K_mm

        # decompose it
        #TODO Warning: kernel matrix os not symmetric after density normalization, eigsh might fail!?
        print('decomposing small kernel matrix') if self.verbose else ''

        lo, hi = K_mm.shape[0]-self.embedding_dim,  K_mm.shape[0]-1  # only calculate the largest ones
        lam_mm, V_mm = eigh(K_mm, eigvals=(lo, hi))
        assert len(lam_mm) == self.embedding_dim, 'sth went wrong with the number of eigenvalues'
        # eigh returns the k largest eigenvals but ascending order (smallest first), so resort
        ix_eig = lam_mm.argsort()[::-1]
        V_mm, lam_mm = V_mm[:,ix_eig], lam_mm[ix_eig]


        # eq. (8,9) in the paper
        if self.verbose:
            print('calculating K_mn')

        # from that, calculate the new eigenvectors, eigenvalues for the big matrix
        K_nm =  self._calc_kernel_mat(X, X[ix,:])  # link to the remaining datapoints

        # approx eigenvectors/eigenvals of the full matrix are just rescaled versions of the small ones
        if self.verbose:
            print('calculating full Eigenvectors')

        # TODO:projection issue
        """
        we shouldnt actually project the subsampled points again?! if m=N we should get the full eigenvectors exaclty
        - it should work for m==n, since  (defintiion of eigenvectors)
                   np.dot(Knm * Vnn) = lam_mm * Vnn
           which nicely cancels all other terms

        - problem is: K_mm is normalized, diagonal zero (diag doesnt matter though, Knm has no 'datapoint onto itself' elements).
           Knm is not normalized! hence the eigenvector Eq doesnt work
          -> proof of that: if we turn off density normalization, it works perfectly
        """

        M = self.m_nystroem
        lam_nxn = (N/M) * lam_mm
        V_nxn = (np.sqrt(M/N)/lam_mm) * np.dot(K_nm, V_mm)

        # import pdb
        # pdb.set_trace()

        # an approximattion Khat to K_full is constructed using (see text between eq7 and eq8)
        # Khat = V_nxn lam_nxn V_nxn.T
        # Khat = np.dot(np.dot(V_nxn, np.diag(lam_nxn)), V_nxn.T )

        return V_nxn, lam_nxn

    def _calc_kernel_mat(self, X, Y=None):
        """
        calculates the kernel matrix. If only X is given, returns the square matrix of X vs
        if X,Y is given, calculates the kernel bewteen those two sets
        :param X:
        :param Y:
        :return:
        """
        if Y is None:
            d = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))
        else:
            d = scipy.spatial.distance.cdist(X, Y)

        return np.exp(-(0.5/self.sigma**2) * d**2)

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
