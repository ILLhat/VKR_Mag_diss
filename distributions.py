from timer import timeit
import numpy as np
from scipy.stats.qmc import Halton


@timeit
def uniform_pos_3d(n_particles: int) -> tuple:
    """
    Generates a uniform positions distribution for particles in 3D space.

    Features:
    1. Perfectly structured initialization
    2. Reproducible results
    3. Low computational cost
    4. Introduce Artificial numerical artifacts (aliasing)
    5. Poor real-world applicability
    """
    # Calculate cubic root, this ensures uniform distribution
    # across 3 space when combining axes, where each dimension
    # contains dim_particles = n_particles**(1/3)
    dim_particles = int(np.cbrt(n_particles))
    # Generate evenly spaced axes over domain length
    x = np.linspace(0, 1, dim_particles)
    y = np.linspace(0, 1, dim_particles)
    z = np.linspace(0, 1, dim_particles)
    # Combine previously generated axes
    return np.meshgrid(x, y, z, indexing="ij")


@timeit
def random_pos_3d(n_particles: int) -> tuple:
    """
    Generates a random positions distribution for particles in 3D space
    based on uniform distibution.

    Features:
    1. Natural noise introduction
    2. Good for Monte Carlo-like methods
    3. Requires more particles for smoothness
    4. More physically realistic
    """
    # Generate uniformly distributed axes over domain length
    x = np.random.uniform(0, 1, n_particles)
    y = np.random.uniform(0, 1, n_particles)
    z = np.random.uniform(0, 1, n_particles)
    return x, y, z


@timeit
def halton_pos_3d(n_particles: int) -> tuple:
    """
    Generates a quasi-random positions distribution for particles in 3D space.
    This function uses scipy implementation, it's 3.5 times faster than pure python
    on 1000000 n_particles.

    Features:
    1. Low-discrepancy sequence
    2. Prevents clustering artifacts
    3. Optimal space filling
    5. Higher computational cost
    """
    halton_sampler = Halton(3, scramble=True).fast_forward(1)
    sample = halton_sampler.random(n_particles)
    return sample[:, 0], sample[:, 1], sample[:, 2]


@timeit
def maxwellian_vel_3d(
    n_particles: int, charge: float, temperature: float, mass: float
) -> tuple:
    """
    Generates Maxwellian velocities for particles in 3D space.
    """
    # Use most probable speed thermal velocity formula
    v_th = np.sqrt((2 * charge * temperature) / mass)
    # Generate Gaussian distributed axes over
    # drift(0) and thermal velocity
    vx = np.random.normal(0, v_th, n_particles)
    vy = np.random.normal(0, v_th, n_particles)
    vz = np.random.normal(0, v_th, n_particles)
    return vx, vy, vz


# I DON'T THINK THERE IS ANY LOGIN IN DOING KAPPA DISTRIBUTION, BECAUSE IT'S NOT FOR DEFAULT PLASMA
