import pytest
import numpy as np
from diffusionMaps import DiffusionMap
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
    distances, indices = df.__get_NN__(X,k=kNN)

    assert distances.shape == (X.shape[0], kNN)
    assert indices.shape == (X.shape[0], kNN)


def test_DiffusionMap_get_kernel_matrix_number_of_neighbours(mixtureNormals):
    """actually we would like to test for the exact number of neighvours
    but due tot the symmetrizing, it can exceed the kNN"""
    X = mixtureNormals
    embDim = 2
    df = DiffusionMap(sigma=1, embedding_dim=embDim)

    kNN = 4
    K = df.__get_kernel_matrix__(X,k=kNN)
    assert K.shape == (X.shape[0], X.shape[0])

    nonzero_elements_per_row = np.sum(K.toarray()!=0, 1)
    print(nonzero_elements_per_row)
    assert np.all(nonzero_elements_per_row >= kNN)  # the number of nonzero elements must be kNN or larger (due to the symmetrizing


def test_DiffusionMap_get_kernel_matrix_symmetry(mixtureNormals):
    "make sure the kernel matrix is symmetric"
    X = mixtureNormals
    df = DiffusionMap(sigma=1,embedding_dim=2,verbose=False)
    K = df.__get_kernel_matrix__(X,k=2)

    Q = (K-K.T).toarray()  # np.all doesnt work on sparse matrices
    assert np.all(Q==0), 'returned kernel matrix is not symmetric'


def test__get_kernel_matrix_sparse(mixtureNormals):
    df = DiffusionMap(sigma=1,embedding_dim=2,verbose=False)
    K = df.__get_kernel_matrix__(mixtureNormals,k=10)
    assert issparse(K)


def test__density_normalize__sparse(mixtureNormals):

    df = DiffusionMap(sigma=1,embedding_dim=2,verbose=False)
    K = csr_matrix([[1,1],[0,1]])
    assert issparse(df.__density_normalize__(K)), 'returned matrix is not sparse after normalization'

def test__density_normalize__colsum(mixtureNormals):

    df = DiffusionMap(sigma=1,embedding_dim=2,verbose=False)
    K = scipy.sparse.rand(100,100,density=0.1)
    K_norm = df.__density_normalize__(K)

    np.testing.assert_allclose(K_norm.toarray().sum(0), 1)

def test_DiffusionMap_fit_transform_eigenvalue_ordering(mixtureNormals):
    "must return the largest first"
    X = mixtureNormals

    embDim = 2
    df = DiffusionMap(sigma=1, embedding_dim=embDim )
    X_embed, lam = df.fit_transform(X)
    assert(lam[0]> lam[1])


def test_DiffusionMap_density_normalization(mixtureNormals):

    df = DiffusionMap(sigma=1, embedding_dim=2 )

    A = np.random.normal(0,1,size=(10,2))
    K = np.dot(A,A.T)  # 10x10 symmetric kernel-matrix
    K = csr_matrix(K)

    K_norm = df.__density_normalize__(K)
    theSum = K_norm.sum(0)
    np.testing.assert_allclose(theSum, 1)
