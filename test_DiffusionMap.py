import pytest
import numpy as np
from diffusionMaps import DiffusionMap, _density_normalize
from scipy.sparse import csr_matrix, issparse
import scipy.sparse


@pytest.fixture
def mixtureNormals(request):
    # teardown
    def fin():
        pass
    request.addfinalizer(fin)

    samples = 50
    dims = 3
    X1 = np.random.normal(0,1, size=(samples,dims))
    X2 = np.random.normal(3,1, size=(samples, dims))

    X = np.vstack([X1,X2])
    return X

def create_sym_matrix(n):
    q = np.random.rand(n,n)
    return np.dot(q,q.T)


def create_sparse_sym_matrix(n, density=0.1): # sparseness = .9 -> 90% entries zero
    q = np.random.rand(n,n)
    sym = np.dot(q,q.T)
    ix = np.random.binomial(1,density/2, size=(n,n))
    ix = ix+ix.T*(ix==0)  # make the selection of entrys also symmetric
    return csr_matrix(sym*ix)


def test_DiffusionMap_fit_transform_output_dimensions(mixtureNormals):
    X = mixtureNormals

    embDim = 2
    df = DiffusionMap(sigma=1, embedding_dim=embDim )
    X_embed, lam = df.fit_transform(X)

    assert X_embed.shape == (X.shape[0], embDim ), "returns wrong dimensionally"
    assert lam.shape[0] == X_embed.shape[1], "must return as many eigenvalues as embedded dimensions"


def test_DiffusionMap_nearestNeighbour_number_of_neighbours(mixtureNormals):
    X = mixtureNormals
    embDim = 2
    df = DiffusionMap(sigma=1, embedding_dim=embDim)

    kNN = 4
    distances, indices = df._get_NN(X,k=kNN)

    assert distances.shape == (X.shape[0], kNN)
    assert indices.shape == (X.shape[0], kNN)


def test_DiffusionMap_get_kernel_matrix_number_of_neighbours(mixtureNormals):
    """actually we would like to test for the exact number of neighvours
    but due tot the symmetrizing, it can exceed the kNN"""
    X = mixtureNormals
    embDim = 2
    df = DiffusionMap(sigma=1, embedding_dim=embDim)

    kNN = 4
    K = df._get_kernel_matrix(X,k=kNN)
    assert K.shape == (X.shape[0], X.shape[0])

    nonzero_elements_per_row = np.sum(K.toarray()!=0, 1)
    print(nonzero_elements_per_row)
    assert np.all(nonzero_elements_per_row >= kNN)  # the number of nonzero elements must be kNN or larger (due to the symmetrizing


def test_DiffusionMap_get_kernel_matrix_symmetry(mixtureNormals):
    "make sure the kernel matrix is symmetric"
    X = mixtureNormals
    df = DiffusionMap(sigma=1,embedding_dim=2)
    K = df._get_kernel_matrix(X,k=2)

    Q = (K-K.T).toarray()  # np.all doesnt work on sparse matrices
    assert np.all(Q==0), 'returned kernel matrix is not symmetric'


def test__get_kernel_matrix_sparse(mixtureNormals):
    df = DiffusionMap(sigma=1,embedding_dim=2)
    K = df._get_kernel_matrix(mixtureNormals,k=10)
    assert issparse(K)


def test__density_normalize__sparse(mixtureNormals):
    "must return sparse if we put in sparse"
    K = csr_matrix([[0,1],[1,1]])
    assert issparse(_density_normalize(K)), 'returned matrix is not sparse after normalization'


def test__density_normalize__rowsum(mixtureNormals):
    "enforce rows summing to on for the desniy normalization"
    K = create_sparse_sym_matrix(100, density=0.1)
    K_norm = _density_normalize(K)
    np.testing.assert_allclose(K_norm.toarray().sum(1), 1)


def test__density_normalize__not_sparse_rowsum(mixtureNormals):
    "enforce rows summing to on for the desniy normalization"
    K = create_sym_matrix(100)
    K_norm = _density_normalize(K)
    np.testing.assert_allclose(K_norm.sum(1), 1)


# def test__density_normalize__not_sparse_symmetrize(mixtureNormals):
#     "check the symmetrized version of the transtion matrix"
#     K = create_sym_matrix(100)
#     Tsym = _density_normalize(K, symmetrize=True)
#     # check symmetry
#     np.testing.assert_allclose(Tsym, Tsym.T, err_msg="Tsym is not symmetric")
#     # check the rowsum =1
#     np.testing.assert_allclose(Tsym.sum(1), 1)

def test__density_normalize__sparse_symmetrize(mixtureNormals):
    "check the symmetrized version of the transtion matrix"
    K = create_sparse_sym_matrix(100, 0.1)
    Tsym = _density_normalize(K, symmetrize=True)
    # check symmetry
    np.testing.assert_allclose(Tsym.toarray(), Tsym.T.toarray(), err_msg="Tsym is not symmetric")
    # check the rowsum =1
    np.testing.assert_allclose(Tsym.toarray().sum(1), 1, err_msg="rowsum <> 1")

def test__density_normalize__not_sparse(mixtureNormals):
    K = create_sym_matrix(2)
    K_norm = _density_normalize(K)
    assert isinstance(K_norm, np.ndarray), 'must return full matrix if we put in a full matrix'



def test_density_normalize_same_result_sparse_nonsparse():

    for d in [0.0001, 0.01, 0.5]:  # test it for different sparsity, as sometimes entre rows/col become zero
        K_sparse = create_sparse_sym_matrix(5, density=d)
        K_full = K_sparse.A

        n_sparse = _density_normalize(K_sparse).A
        n_full = _density_normalize(K_full)

        np.testing.assert_allclose(n_sparse, n_full)


def test_DiffusionMap_fit_transform_eigenvalue_ordering(mixtureNormals):
    "must return the largest first"
    X = mixtureNormals

    embDim = 2
    df = DiffusionMap(sigma=1, embedding_dim=embDim )
    X_embed, lam = df.fit_transform(X)
    assert(lam[0]> lam[1])
