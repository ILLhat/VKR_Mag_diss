from distributions import (
    uniform_pos_3d,
    random_pos_3d,
    halton_pos_3d,
    maxwellian_vel_3d,
)
from depositions import (
    nearest_grid_point_deposit,
    cloud_in_cell_deposit,
    triangular_shaped_cloud_deposit,
)
from efield_solvers import poisson_fft
from gather import triangular_shaped_cloud_gather, cloud_in_cell_gather
from particles_motion import boris_push

if __name__ == "__main__":
    # Parameters
    n_particles = 10000
    charge = 1.602176634e-19
    mass = 9.1093837015e-31
    c_over_m = charge / mass
    temperature = 10
    grid_size = (64, 64, 64)
    cell_size = 1 / 64
    dt = 1e-18
    # Create uniform position distibution
    uniform_pos = uniform_pos_3d(n_particles)
    # Create random position distribution
    random_pos = random_pos_3d(n_particles)
    # Create quasi-random position distibution
    halton_pos = halton_pos_3d(n_particles)
    # Create Maxwellian velocity distribution
    vel = maxwellian_vel_3d(n_particles, charge, temperature, mass)
    # Deposit our halton_pos with nearest grid point method
    ngp_charge_density = nearest_grid_point_deposit(
        halton_pos, n_particles, grid_size, cell_size, -charge
    )
    # Deposit our halton_pos with nearest cloud in cell method
    cic_charge_density = cloud_in_cell_deposit(
        halton_pos, n_particles, grid_size, cell_size, -charge
    )
    # Deposit our halton_pos with nearest triangular shaped cloud method
    tcs_charge_density = triangular_shaped_cloud_deposit(
        halton_pos, n_particles, grid_size, cell_size, -charge
    )
    method = "cic"
    if method == "tcs":
        # Compute Poisson equation to get efield on grid
        grid_efield = poisson_fft(tcs_charge_density, grid_size, cell_size)
        # Get particle efield by gathering efield from grid to particle
        efield = triangular_shaped_cloud_gather(
            grid_efield, halton_pos, n_particles, grid_size, cell_size, -charge
        )
        # Push particles using boris push
        halton_pos, vel = boris_push(
            halton_pos, vel, efield, n_particles, c_over_m, dt
        )
    elif method == "cic":
        # Compute Poisson equation to get efield on grid
        grid_efield = poisson_fft(cic_charge_density, grid_size, cell_size)
        # Get particle efield by gathering efield from grid to particle
        efield = cloud_in_cell_gather(
            grid_efield, halton_pos, n_particles, grid_size, cell_size, -charge
        )
        # Push particles using boris push
        halton_pos, vel = boris_push(
            halton_pos, vel, efield, n_particles, c_over_m, dt
        )
