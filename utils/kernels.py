import numpy as np
import matplotlib.pyplot as plt
import torch


def invert_matrix(K, method='cholesky'):
    """ Invert a given matrix using the Cholesky decomposition or SVD. """

    if method == 'cholesky':
        L = compute_cholesky(K, eps=1e-6)
        L_inv = np.linalg.inv(L)
        return L_inv.dot(L_inv.T)
    
    elif method == 'svd':        
        U_times_sqrt_S, (U, S, Vt) = compute_svd_factor(K, eps=1e-6)        
        S_inv = np.diag(1/S)
        return U.dot(S_inv).dot(U.T)

    elif method == 'numpy':
        return np.linalg.inv(K)
    
    else:
        raise ValueError(f"Invalid method: {method}")


def inverse_square_root(K, method='cholesky'):
    """ Compute the inverse square root of a given matrix using the Cholesky decomposition or SVD. """

    if method == 'cholesky':
        L = compute_cholesky(K, eps=1e-6)
        L_inv = np.linalg.inv(L)
        return L_inv
    
    elif method == 'svd':
        U_times_sqrt_S, (U, S, Vt) = compute_svd_factor(K, eps=1e-6)
        U_times_sqrt_S_inv = np.diag(1/U_times_sqrt_S)
        return U.dot(U_times_sqrt_S_inv)
    
    else:
        raise ValueError(f"Invalid method: {method}")


def whiten_data(data_samples, K, method='cholesky'):
    """Whiten the data using the given covariance matrix."""

    L_inv = inverse_square_root(K, method=method)

    mean_data = np.mean(data_samples, axis=0)
    whitened_data = (data_samples - mean_data) @ L_inv.T

    return whitened_data


def whiten_data_kronecker(data_samples, mean_data, K1, K2, method='cholesky'):
    """Whiten the data using the Kronecker product of two covariance matrices."""
    
    if type(K1) == torch.Tensor:
        return_pytorch = True
        K1 = K1.numpy()
        K2 = K2.numpy()
        data_samples = data_samples.numpy()
        mean_data = mean_data.numpy()

    Lk1  = inverse_square_root(K1, method=method)
    Lk2  = inverse_square_root(K2, method=method)

    # mean_data = np.mean(data_samples, axis=0)
    shifted_data = data_samples - mean_data

    a = np.einsum('ij,njk->nik', Lk1, shifted_data)
    whitened_data = np.einsum('nik,kl->nil', a, Lk2.T)    

    if return_pytorch:
        return torch.tensor(whitened_data).float()

    return whitened_data


def sample_from_covariance(K, num_samples=10, method='cholesky'):
    """
    Sample from a given covariance matrix.
    """
    if method == 'cholesky':
        return sample_cholesky(K, num_samples)
    elif method == 'svd':
        return sample_svd(K, num_samples)
    elif method == 'numpy':
        return np.random.multivariate_normal(np.zeros(K.shape[0]), K, num_samples)
    else:
        raise ValueError(f"Invalid method: {method}")


def sample_from_kronecker_covariance(K1, K2, num_samples=10, method='cholesky'):    
    """Sample from the Kronecker product of two covariance matrices using the mixed-product properties."""

    dim_k1 = K1.shape[0]
    dim_k2 = K2.shape[0]
    normal_samples = np.random.randn(num_samples, dim_k1 * dim_k2)

    if method == 'cholesky':
        Lk1  = compute_cholesky(K1, eps=1e-6)
        Lk2  = compute_cholesky(K2, eps=1e-6)

        test_sample = normal_samples.reshape(num_samples, dim_k1, dim_k2)
        a = np.einsum('ij,njk->nik', Lk1, test_sample)
        return np.einsum('nik,kl->nil', a, Lk2.T)
    
    elif method == 'svd':
        U_times_sqrt_S1, _ = compute_svd_factor(K1, eps=1e-6)
        U_times_sqrt_S2, _ = compute_svd_factor(K2, eps=1e-6)

        test_sample = normal_samples.reshape(num_samples, dim_k1, dim_k2)
        a = np.einsum('ij,njk->nik', U_times_sqrt_S1, test_sample)
        return np.einsum('nik,kl->nil', a, U_times_sqrt_S2.T)        
    
    else:
        raise ValueError(f"Invalid method: {method}")


def sample_cholesky(K, num_samples=10):
    """
    Sample from a given covariance matrix.
    """
    # Compute Cholesky decomposition of the covariance matrix
    L = compute_cholesky(K, eps=1e-6)
    
    # Generate samples from standard normal distribution
    normal_samples = np.random.randn(num_samples, L.shape[0])
    
    # Transform samples to have desired covariance structure
    samples = np.dot(normal_samples, L.T)
    
    return samples


def sample_svd(K, num_samples=10):
    """
    Sample from a given covariance matrix using SVD.
    """
    U_times_sqrt_S, (U, S, Vt) = compute_svd_factor(K, eps=1e-6)
    
    # Generate samples from standard normal distribution
    normal_samples = np.random.randn(num_samples, K.shape[0])
    
    # Compute sampled data    )
    samples = np.dot(normal_samples, U_times_sqrt_S.T)
    
    return samples


def compute_cholesky(K, eps=1e-6):
    """
    Compute Cholesky decomposition of a given covariance matrix.
    """
    # Compute Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(K + eps*np.eye(K.shape[0]))
    
    return L


def compute_svd_factor(K, eps=1e-6):
    # Compute SVD of the covariance matrix
    U, S, Vt = np.linalg.svd(K, full_matrices=True)
    
    # Regularize singular values if necessary    
    S_reg = np.maximum(S, eps)  # ensure positive values
    
    # Compute square root of regularized singular values
    sqrt_S = np.sqrt(S_reg)
    
    # Construct matrix U * sqrt(S)
    U_times_sqrt_S = np.dot(U, np.diag(sqrt_S))

    return U_times_sqrt_S, (U, S, Vt)


def quadratic_kernel(x: np.ndarray,
                     x_prime: np.ndarray,
                     lengthscale: float, 
                     sigma: float,
                     ):
    """Generate a quadratic kernel of length 'length' and standard deviation 'sigma'."""

    assert x.shape == x_prime.T.shape, "x and x_prime must have the same shape"
    assert lengthscale > 0, "lengthscale must be positive"
    assert sigma > 0, "sigma must be positive"
    if x.ndim == 1:
        x = x[:, None]
        x_prime = x_prime[:, None]
        x_prime = x_prime.T
    
    l2_norm = ((x - x_prime)**2)
    K = sigma**2 * np.exp(-l2_norm / 2*lengthscale**2)
    
    return K
    

def periodic_kernel(x: np.ndarray,
                    x_prime: np.ndarray,
                    lengthscale: float, 
                    sigma: float,
                    p: float):
    """Generate a periodc kernel of length 'length' and standard deviation 'sigma' and period p."""

    assert x.shape == x_prime.T.shape, "x and x_prime must have the same shape"
    assert lengthscale > 0, "lengthscale must be positive"
    assert sigma > 0, "sigma must be positive"
    assert p > 0, "p must be positive"
    if x.ndim == 1:
        x = x[:, None]
        x_prime = x_prime[:, None]
        x_prime = x_prime.T

    l_norm = np.abs(x - x_prime)
    K = sigma**2 * np.exp(-2 / lengthscale**2 * np.sin(np.pi * l_norm / p)**2)
    
    return K


class Kernel(object):
    def __init__(self, kernel, eps=1e-6):
        self.kernel = kernel
        self.eps = eps  # For the numerical stability of the Cholesky decomposition and SVD
    
    def plot_kernel(self, num_samples = 10):
        # Draw samples from the quadratic kernel
        np.random.seed(0)

        x = np.linspace(0, 10, 100)[:, None]
        K, dis = self.__call__(x, x.T, return_dis=True)

        # Flatten the kernel and distance matrices, and sort by distance
        K_flat = K.flatten()
        dis_flat = dis.flatten()
        sort_idx = np.argsort(dis_flat)
        dis_flat = dis_flat[sort_idx]
        K_flat = K_flat[sort_idx]

        # Draw N samples from the quadratic kernel        
        samples = np.random.multivariate_normal(np.zeros(x.shape[0]), K, num_samples)

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].imshow(K); ax[0].set_title("Kernel Covariance"); ax[0].set_xlabel("x"); ax[0].set_ylabel("x")
        ax[1].plot(dis_flat, K_flat); ax[1].set_title("Kernel Profile"); ax[1].set_xlabel("Distance"); ax[1].set_ylabel("Kernel Value")
        ax[2].plot(x, samples.T); ax[2].set_title("Kernel Samples"); ax[2].set_xlabel("x"); ax[2].set_ylabel("f(x)")
        plt.show()

    def __call__(self, x, x_prime, return_dis=False):
        """Compute the kernel matrix. - It follows the scheme below"""
        K = self.kernel(x, x_prime)
        self.L = np.linalg.cholesky(K + self.eps*np.eye(K.shape[0]))
        self.U_sqrtS, self.SVD = compute_svd_factor(K, eps=1e-6)
        if return_dis:
            return K, (x - x_prime)
        return K


class QuadraticKernel(Kernel):
    def __init__(self, lengthscale, sigma):
        super().__init__(quadratic_kernel)
        self.lengthscale = lengthscale
        self.sigma = sigma

    def __call__(self, x, x_prime, return_dis=False):
        K = self.kernel(x, x_prime, self.lengthscale, self.sigma)
        self.L = np.linalg.cholesky(K + self.eps*np.eye(K.shape[0]))
        self.U_sqrtS, self.SVD = compute_svd_factor(K, eps=1e-6)
        if return_dis:
            return K, (x - x_prime)
        return K
    

class PeriodicKernel(Kernel):
    def __init__(self, lengthscale, sigma, p):
        super().__init__(periodic_kernel)
        self.lengthscale = lengthscale
        self.sigma = sigma
        self.p = p

    def __call__(self, x, x_prime, return_dis=False):
        K = self.kernel(x, x_prime, self.lengthscale, self.sigma, self.p)
        self.L = np.linalg.cholesky(K + self.eps*np.eye(K.shape[0]))
        self.U_sqrtS, self.SVD = compute_svd_factor(K, eps=1e-6)
        if return_dis:
            return K, (x - x_prime)
        return K


class LocalPeriodicKernel(Kernel):
    def __init__(self, lengthscale_spatial, lengthscale_temporal, sigma_spatial, sigma_temporal, p):
        self.quadratic_kernel = QuadraticKernel(lengthscale_spatial, sigma_spatial)
        self.periodic_kernel = PeriodicKernel(lengthscale_temporal, sigma_temporal, p)
        super().__init__(lambda x, y: self.quadratic_kernel(x, y) * self.periodic_kernel(x, y))

        self.lengthscale_spatial = lengthscale_spatial
        self.lengthscale_temporal = lengthscale_temporal
        self.sigma_spatial = sigma_spatial
        self.sigma_temporal = sigma_temporal
        self.p = p
    
    def __call__(self, x, x_prime, return_dis=False):
        # K = self.quadratic_kernel(x, x_prime) * self.periodic_kernel(x, x_prime)
        K = self.kernel(x, x_prime)
        self.L = np.linalg.cholesky(K + self.eps*np.eye(K.shape[0]))
        self.U_sqrtS, self.SVD = compute_svd_factor(K, eps=1e-6)
        if return_dis:
            return K, (x - x_prime)
        return K


if __name__ == "__main__":
    # print('he')
    rbf_kernel = QuadraticKernel(1, 1)
    per_kernel = PeriodicKernel(1, 1, 1)
    local_periodic_kernel = LocalPeriodicKernel(1, 1, 1, 1, 1)

    x = np.linspace(0, 10, 100)[:, None]
    Krbf = rbf_kernel(x, x.T, return_dis=False)
    Kper = per_kernel(x, x.T, return_dis=False)
    Klocal = local_periodic_kernel(x, x.T, return_dis=False)

    # ======================== Plot the different covariances ========================
    # fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    # ax[0].imshow(Krbf); ax[0].set_title("Quadratic Kernel Covariance"); ax[0].set_xlabel("x"); ax[0].set_ylabel("x")
    # ax[1].imshow(Kper); ax[1].set_title("Periodic Kernel Covariance"); ax[1].set_xlabel("x"); ax[1].set_ylabel("x")
    # ax[2].imshow(Klocal); ax[2].set_title("Local Periodic Kernel Covariance"); ax[2].set_xlabel("x"); ax[2].set_ylabel("x")

    # =========================== Plot the different kernels ===========================
    rbf_kernel.plot_kernel(num_samples = 10)