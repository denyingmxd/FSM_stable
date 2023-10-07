import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import data, util
from sklearn.feature_extraction import image
import scipy as sp
import tqdm
from scipy.ndimage import zoom

import multiprocessing
# params for image and noise
sigma = 0.1                     # noise standard deviation
          # patch dimensions
             # number of pixels per patch
C = 1.15                        # noise gain
resize=False
if resize:
    patch_dims = (4, 4)
else:
    patch_dims = (8, 8)
n = np.prod(patch_dims)
# params KSVD, see the KSVD class in the next cell
k = 32                          # number of dictionary atoms
maxiter = 5                     # number of KSVD iterations
omp_tol = n * (sigma * C)**2    # OMP tolerance defined in [1]
omp_nnz = None                  # OMP non-zero coeffs targeted over the n coeffs
param_a = 0.5                   # parameter for the modified scalar-product

# !wget -O house.tiff https://tinyurl.com/yx99dmu2 > /dev/null 2>&1
# Read image and add noise
img = mpimg.imread("./house.tiff")
img = util.img_as_float(img)
if resize:
    img = zoom(img,(0.125,0.125,1))

img_with_noise = util.random_noise(img, var=sigma**2)

# Extract and concatenate patches to obtain the input
patches = image.extract_patches_2d(img, patch_dims)
Y = patches.reshape(patches.shape[0], -1)

#### Visualize the original image and the corrupted image
# plt.figure(figsize=(14,7))
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(img_with_noise)
# plt.show()

class K_SVD:
    """
    Implementation of K-SVD algorithm discribed in [1]. The truncated SVD
    operation is approximated using the algorithm in [2].

    References :
    ------------
    [1] 'Sparse Representation for Color Image Restoration. J. Mairal,
        M. Elad, G. Sapiro'.
    [2] 'Efficient Implementation of the K-SVD Algorithm using Batch
        Orthogonal Matching Pursuit. R. Rubinstein, M. Zibulevsky and M. Elad'.

    """

    def __init__(self, k, maxiter, omp_tol=None, omp_nnz=None, param_a=0.):
        """
        Parameters
        ----------
        k : int
            # of dictionary atoms
        maxiter : int
            Maximum number of iterations
        omp_tol : float
            Desired precision in OMP
        omp_nnz : int
            Target number of non-zero coefficients in OMP
        param_a : float
           Correction parameter a (where gamma = 2a+a^2) for the modified scalar
           product.
        """
        self.k = k
        self.maxiter = maxiter
        self.omp_tol = omp_tol
        self.omp_nnz = omp_nnz
        self.dictionary = None
        self.param_a = param_a
        self.modified_metric_matrix = None

        if omp_tol is None and omp_nnz is None:
            raise ValueError("Either omp_tol or omp_nnz must be specified.")

    def _initialize(self, Y):
        if min(Y.shape) < self.k:
            D = np.random.randn(self.k, Y.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(Y, k=self.k)
            D = np.diag(s) @ vt
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]

        # Metric modification matrix
        n = D.shape[1]
        I = np.eye(n)
        J = np.ones((int(n / 3), int(n / 3)))
        K = sp.linalg.block_diag(J, J, J)
        self.modified_metric_matrix = I + (self.param_a / n) * K

        return D

    def _update_dict(self, Y, D, alpha):
        for j in range(self.k):
            I = alpha[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = alpha[I, j].T
            E = Y[I, :] - alpha[I, :] @ D
            d = E.T @ g
            d /= np.linalg.norm(d)
            g = E @ d
            D[j, :] = d
            alpha[I, j] = g.T
        return D, alpha

    def _denoise(self, D, Y):

        D = D @ self.modified_metric_matrix
        Y = Y @ self.modified_metric_matrix

        return matrix_omp(D.T, Y.T, tol=self.omp_tol, nnz=self.omp_nnz).T

    def dictionary_learning(self, Y):
        """
        Learn a dictionary for a given input signal.

        Parameters
        ----------
        Y : np.ndarray with 2 dims
            Input signal.
        """
        D = self._initialize(Y)
        for i in range(self.maxiter):
            ########### Start your code ############

            ### hint: use the functions that has been
            ###         written in this class
            print(f" ====== OMP step {i + 1} ======")
            alpha = self._denoise(D, Y)
            print(" ====== Dictionary update step ======")
            D, alpha = self._update_dict(Y, D, alpha)

            ########### End your code ############

        self.dictionary = D

    def denoise(self, Y):
        """
        Find the sparse representation of the input signal.

        Parameters
        ----------
        Y : np.ndarray with 2 dims
            Input signal.
        """

        if self.dictionary is None:
            raise ValueError("Dictionary not learned yet, consider `learn_dictionary(Y)` first.")

        print("Denoising signal ...")
        return self._denoise(self.dictionary, Y)


def matrix_omp(D, Y, tol=None, nnz=None):
    """
    Orthogonal Matching Pursuit algorithm when the signal is a matrix.

    Parameters
    ----------
    D : np.ndarray with 2 dims
        Input dictionary. Each colum is a dictionary atom.
    Y : np.array with 2 dims
        Input targets
    nnz : int
        Targeted number of non-zero elements.
    tol : float
        Targeted precision.
    """

    X = np.zeros((D.shape[1], Y.shape[1]))


    for k in tqdm.tqdm(range(Y.shape[1])):
        X_k, idx_k = omp(D, Y[:, k], tol, nnz)

        ########### Start your code ############
        X[idx_k, k] = X_k
        ########### End your code ############

    return X

def omp(D, y, tol=None, nnz=None):
    """
    Orthogonal Matching Pursuit algorithm.

    Parameters
    ----------
    D : np.ndarray with 2 dims
        Input dictionary. Each colum is a dictionary atom.
    y : np.array with 1 dim
        Input targets
    nnz : int
        Targeted number of non-zero elements.
    tol : float
        Targeted precision.
    """

    m, n = D.shape

    if nnz is None and tol is None:
        raise ValueError("Either nnz or tol must be specified.")
    if nnz is not None and tol is not None:
        tol = None
    if nnz is None:
        nnz = n
    elif nnz > n:
        raise ValueError("Parameter nnz exceed A.shape[1].")
    if tol is None:
        tol = 0

    idx = []
    n_active = 0
    res = y

    while True:

        ########### Start your code ############
        ### hint: the same atom cannot be selected twice
        rs = np.abs(D.T @ res)
        k = np.argmax(rs)
        if k in idx:
            break
        n_active += 1
        idx.append(k)
        A = D[:,idx]
        x_idx = np.linalg.inv((A.T@A))@A.T@y
        res = y - D[:,idx] @ x_idx
        # print(np.linalg.norm(res), tol, n_active, nnz)
        ### hint: write the condition to break the loop with 'tol' and 'nnz'
        if np.linalg.norm(res) < tol or n_active >= nnz:
            break
        ########### End your code ############

    return x_idx, idx

# Image denoising with K-SVD

k_svd = K_SVD(k=k, maxiter=maxiter, omp_tol=omp_tol, omp_nnz=None, param_a=param_a)
k_svd.dictionary_learning(Y)
alpha = k_svd.denoise(Y)

img_reconstructed = alpha @ k_svd.dictionary
img_reconstructed = image.reconstruct_from_patches_2d(
    img_reconstructed.reshape(patches.shape), img.shape)

plt.figure(figsize=(30,10))
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(img_with_noise)
plt.subplot(133)
plt.imshow(img_reconstructed)
fig1 = plt.gcf()
if resize:
    fig1.savefig("result_resize.png")
else:
    fig1.savefig("result.png")
# plt.show()