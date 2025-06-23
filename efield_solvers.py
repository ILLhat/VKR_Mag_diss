import numpy as np
from scipy.constants import epsilon_0
from timer import timeit


@timeit
def poisson_fft(rho: np.ndarray, grid_size: float, cell_size: float) -> tuple:
    # Calculate wavevectors and map it to meshgrid
    kx = 2 * np.pi * np.fft.fftfreq(grid_size[0], cell_size)
    ky = 2 * np.pi * np.fft.fftfreq(grid_size[1], cell_size)
    kz = 2 * np.pi * np.fft.fftfreq(grid_size[2], cell_size)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    k_sq = pow(kx, 2) + pow(ky, 2) + pow(kz, 2)
    k_sq[0, 0, 0] = 1
    # Solve Poisson equation in Fourier space
    rho_hat = np.fft.fftn(rho)
    phi_hat = rho_hat / (epsilon_0 * k_sq)
    phi_hat[0, 0, 0] = 0
    # Compute electric field via spectral differentiation
    efield_x_hat = -1j * kx * phi_hat
    efield_y_hat = -1j * ky * phi_hat
    efield_z_hat = -1j * kz * phi_hat
    efield_x = np.real(np.fft.ifftn(efield_x_hat))
    efield_y = np.real(np.fft.ifftn(efield_y_hat))
    efield_z = np.real(np.fft.ifftn(efield_z_hat))
    return (efield_x, efield_y, efield_z)
